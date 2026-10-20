"""Full direct-path decomposition of a weak event's gate pre-activation: embedding and MLP neurons at the same token
(direct), plus attention-mediated paths through every head's OV circuit to MLP neurons and embeddings at every source
position (attention pattern frozen). Produces a unified ranking of upstream neurons by total contribution.
Writes results/<tag>/driver_sample_full.json (events with top-10 unified drivers) and upstream_full_summary.json."""
import argparse, json, os, sys, time, random, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, HIST_LO, HIST_HI, HIST_BINS, cum_table
from data_prep import fineweb_token_stream
from run_grid import ROOT

BANDS = {'b50_80': (50, 80), 'b80_95': (80, 95), 'strong': (99.9, 100.0)}
PRE, POST = 40, 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--n_down', type=int, default=30); ap.add_argument('--per_band', type=int, default=10); ap.add_argument('--tokens', type=int, default=3_000_000)
    ap.add_argument('--topk', type=int, default=10); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
    layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features; D = layers[0].mlp.gate_proj.in_features
    cfg = model.config; H = cfg.num_attention_heads; KV = cfg.num_key_value_heads; hd = getattr(cfg, 'head_dim', D // H); rep = H // KV
    # same downstream sample as driver_sample.json (seeded), restricted to layers >= 2
    prev = json.load(open(os.path.join(rd, 'driver_sample.json')))
    downs = sorted({(s['layer'], s['neuron'], s['wasserstein']) for s in prev})
    sel_by_layer = {}
    for i, (l, n, w) in enumerate(downs): sel_by_layer.setdefault(l, []).append((i, n))
    qs = sorted({b for v in BANDS.values() for b in v if b < 100})
    d, st = thresholds_from_stats(os.path.join(rd, 'stats.pt'), qs); thr = {q: d[q][0][:, :, 0].cpu() for q in qs}
    cum = cum_table(st); w_ = (HIST_HI - HIST_LO) / HIST_BINS
    def pct(l, m, quad, val):
        b = int(min(HIST_BINS - 1, max(0, math.floor((math.log10(max(abs(val), 1e-30)) - HIST_LO) / w_)))); return float(cum[l, m, quad, b])
    cap = {}
    for l, layer in enumerate(layers):
        layer.post_attention_layernorm.register_forward_pre_hook(lambda mod, inp, l=l: cap.__setitem__(('resid', l), inp[0].detach()))
        layer.input_layernorm.register_forward_pre_hook(lambda mod, inp, l=l: cap.__setitem__(('resid1', l), inp[0].detach()))
        layer.self_attn.register_forward_hook(lambda mod, inp, out, l=l: cap.__setitem__(('attn', l), out[0].detach()))
    model.model.embed_tokens.register_forward_hook(lambda mod, inp, out: cap.__setitem__('emb', out.detach()))
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    class Cap:
        def record(self, l, g, u, a): cap[('a', l)] = a.detach(); cap[('quad', l)] = (2 * (g < 0).to(torch.int8) + (u < 0).to(torch.int8))
    interv.stats = Cap()
    Wg = [ly.mlp.gate_proj.weight.detach() for ly in layers]; Wd = [ly.mlp.down_proj.weight.detach().float() for ly in layers]
    g2 = [ly.post_attention_layernorm.weight.detach().float() for ly in layers]; g1 = [ly.input_layernorm.weight.detach().float() for ly in layers]
    Wv = [ly.self_attn.v_proj.weight.detach().float() for ly in layers]; Wo = [ly.self_attn.o_proj.weight.detach().float() for ly in layers]  # Wv [KV*hd, D], Wo [D, H*hd]
    eps = getattr(layers[0].post_attention_layernorm, 'variance_epsilon', 1e-6)
    counts = {i: {b: 0 for b in BANDS} for i in range(len(downs))}; records = []; t0 = time.time(); nt = 0; nb = 0
    with torch.no_grad():
        for batch in fineweb_token_stream(tok, 1024, 2, a.tokens, split='stats2'):
            cap.clear(); x = batch.cuda(); out = model(input_ids=x, use_cache=False, output_attentions=True); attns = out.attentions  # tuple [B,H,T,T]
            nt += batch.numel(); nb += 1; B, T = batch.shape
            for l, lst in sel_by_layer.items():
                A = cap[('a', l)].float(); Q = cap[('quad', l)]
                for (si, n) in lst:
                    if all(counts[si][b] >= a.per_band for b in BANDS): continue
                    av = A[:, :, n]; q1 = (Q[:, :, n] == 0)
                    for bname, (lo, hi) in BANDS.items():
                        if counts[si][bname] >= a.per_band: continue
                        lo_t = float(thr[lo][l, n]); hi_t = float(thr[hi][l, n]) if hi < 100 else float('inf')
                        m = q1 & (av.abs() >= lo_t) & (av.abs() < hi_t); m[:, :8] = False
                        pos = m.nonzero()
                        if pos.numel() == 0: continue
                        b, t = pos[torch.randint(pos.shape[0], (1,), device=pos.device)].tolist()[0]
                        rec = decompose(l, n, b, t, cap, attns, Wg, Wd, g1, g2, Wv, Wo, eps, H, KV, rep, hd, N, D, pct, batch, tok, a.topk)
                        rec.update(layer=l, neuron=n, band=bname, wasserstein=bool(downs[si][2]), act=float(av[b, t])); records.append(rec); counts[si][bname] += 1
            if nb % 25 == 0:
                print(f'{nt/1e6:.2f}M tokens, {len(records)}/{len(downs)*len(BANDS)*a.per_band} events, {time.time()-t0:.0f}s', flush=True)
                json.dump(records, open(os.path.join(rd, 'driver_sample_full.json'), 'w'))
            if all(counts[si][b] >= a.per_band for si in counts for b in BANDS): break
    json.dump(records, open(os.path.join(rd, 'driver_sample_full.json'), 'w'))
    summ = summarize(records); json.dump(summ, open(os.path.join(rd, 'upstream_full_summary.json'), 'w'), indent=1); print(json.dumps(summ, indent=1), flush=True)


def decompose(l, n, b, t, cap, attns, Wg, Wd, g1, g2, Wv, Wo, eps, H, KV, rep, hd, N, D, pct, batch, tok, topk):
    dev = 'cuda'
    x = cap[('resid', l)][b, t].float(); rms = torch.sqrt((x * x).mean() + eps); v = (Wg[l][n].float() * g2[l]) / rms; g = float(v @ x)
    total_direct = torch.zeros(l * N, device=dev); total_attn = torch.zeros(l * N, device=dev)   # per upstream neuron (l'' < l)
    emb_direct = float(v @ cap['emb'][b, t].float()); emb_attn = 0.0; attn_check = 0.0; attn_true = 0.0
    for lp in range(l):
        total_direct[lp * N:(lp + 1) * N] = (v @ Wd[lp]) * cap[('a', lp)][b, t].float()
    # attention-mediated: for each layer l' <= l and head h, source positions s <= t
    for lp in range(l + 1):
        Aw = attns[lp][b, :, t, :t + 1].float()                                   # [H, t+1]
        x1 = cap[('resid1', lp)][b, :t + 1].float(); rms1 = torch.sqrt((x1 * x1).mean(-1, keepdim=True) + eps)   # [t+1,1]
        attn_true += float(v @ cap[('attn', lp)][b, t].float())
        for h in range(H):
            kv = h // rep
            q = Wv[lp][kv * hd:(kv + 1) * hd].T @ (Wo[lp][:, h * hd:(h + 1) * hd].T @ v)   # [D]: v . Wo_h Wv_kv (.) = q . (.)
            qg = q * g1[lp]                                                          # includes input layernorm gain
            w = Aw[h] / rms1[:, 0]                                                   # [t+1]
            attn_check += float((w * (x1 @ qg)).sum())                               # = v . attn_out via this head (sanity)
            emb_attn += float((w * (cap['emb'][b, :t + 1].float() @ qg)).sum())
            for lpp in range(lp):                                                   # neurons at source positions that wrote into x1
                proj = qg @ Wd[lpp]                                                  # [N]
                wa = w @ cap[('a', lpp)][b, :t + 1].float()                          # [N]
                total_attn[lpp * N:(lpp + 1) * N] += proj * wa
    tot = total_direct + total_attn
    srt = torch.sort(tot, descending=True); top = []
    for val, idx in zip(srt.values[:topk].tolist(), srt.indices[:topk].tolist()):
        lpp, m = divmod(idx, N); am = float(cap[('a', lpp)][b, t, m]); quad = int(cap[('quad', lpp)][b, t, m])
        top.append(dict(layer=lpp, neuron=m, contrib=val, direct=float(total_direct[idx]), via_attn=float(total_attn[idx]), a=am, own_pct=pct(lpp, m, quad, am), quad=quad))
    pos = tot[tot > 0]
    ids = batch[b, max(0, t - PRE):t + 1 + POST].tolist(); tp = t - max(0, t - PRE)
    return dict(g=g, attn_true=attn_true, attn_check=attn_check, emb_direct=emb_direct, emb_attn=emb_attn,
                sum_neurons_direct=float(total_direct.sum()), sum_neurons_attn=float(total_attn.sum()), pos_total=float(pos.sum()),
                top3_share_g=float(srt.values[:3].sum()) / g if g else None, top10_share_g=float(srt.values[:10].sum()) / g if g else None,
                top10_share_pos=float(srt.values[:10].sum() / pos.sum()) if pos.numel() else None, participation=float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.numel() else None,
                top=top, text=dict(before=tok.decode(ids[:tp]), token=tok.decode(ids[tp:tp + 1]), after=tok.decode(ids[tp + 1:])))


def summarize(records):
    import statistics as S; out = {}
    for band in BANDS:
        rs = [r for r in records if r['band'] == band and r['g'] > 0]
        if not rs: continue
        out[band] = dict(n=len(rs), attn_decomp_rel_err=S.median(abs(r['attn_check'] - r['attn_true']) / max(abs(r['attn_true']), 1e-6) for r in rs),
                         total_rel_err=S.median(abs(r['emb_direct'] + r['emb_attn'] + r['sum_neurons_direct'] + r['sum_neurons_attn'] - r['g']) / max(abs(r['g']), 1e-6) for r in rs),
                         median_top3_share_g=S.median(r['top3_share_g'] for r in rs), median_top10_share_g=S.median(r['top10_share_g'] for r in rs),
                         median_top10_share_pos=S.median(r['top10_share_pos'] for r in rs), median_participation=S.median(r['participation'] for r in rs),
                         frac_top10_mostly_via_attn=S.mean(t['via_attn'] > t['direct'] for r in rs for t in r['top']),
                         median_attn_fraction_of_g=S.median(r['sum_neurons_attn'] / r['g'] for r in rs), median_emb_fraction_of_g=S.median((r['emb_direct'] + r['emb_attn']) / r['g'] for r in rs))
    return out


if __name__ == '__main__':
    main()
