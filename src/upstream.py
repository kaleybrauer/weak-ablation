"""Experiment: upstream concentration of weak events.
For sampled Q1 events of selected neurons in percentile bands, decompose the gate pre-activation g into direct
contributions from the embedding, each earlier layer's attention output, and every earlier-layer MLP neuron
(exact given the RMSNorm scale at that token). Report concentration of positive contributions and the own-percentile
of the top upstream neuron drivers. Usage: python upstream.py --model ... --tag ... --n_neurons 120 --tokens 3000000"""
import argparse, json, os, sys, time, random, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, HIST_LO, HIST_HI, HIST_BINS
from data_prep import fineweb_token_stream
from run_grid import ROOT

BANDS = {'b50_80': (50, 80), 'b80_95': (80, 95), 'b95_99': (95, 99), 'strong': (99.9, 100.0)}
PER_BAND = 24; PER_BATCH_CAP = 2; PRE = 40; POST = 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--n_neurons', type=int, default=120); ap.add_argument('--tokens', type=int, default=3_000_000)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--suffix', default=''); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    D = layers[0].mlp.gate_proj.in_features
    rng = random.Random(0)
    wmask = torch.load(os.path.join(rd, 'stats_wasserstein.pt'))['mask']
    widx = [tuple(x) for x in wmask.nonzero().tolist() if x[0] >= 2]; rng.shuffle(widx)
    sel = [(rng.randrange(2, L - 1), rng.randrange(N)) for _ in range(a.n_neurons // 2)] + widx[: a.n_neurons - a.n_neurons // 2]
    sel_by_layer = {}
    for i, (l, n) in enumerate(sel): sel_by_layer.setdefault(l, []).append((i, n))
    qs = sorted({b for v in BANDS.values() for b in v if b < 100})
    d, st = thresholds_from_stats(os.path.join(rd, 'stats.pt'), qs)
    thr = {q: d[q][0][:, :, 0].cpu() for q in qs}  # Q1 thresholds [L,N]
    # cumulative fraction per (layer, neuron, quadrant, bin) for own-percentile lookups
    count = st['count'].double(); cum = (count.cumsum(-1) / count.sum(-1, keepdim=True).clamp_min(1)).half().cuda()  # [L,N,4,B]
    w = (HIST_HI - HIST_LO) / HIST_BINS
    def pct(l, m, quad, val):
        b = int(min(HIST_BINS - 1, max(0, math.floor((math.log10(max(abs(val), 1e-30)) - HIST_LO) / w))))
        return float(cum[l, m, quad, b])
    # capture buffers
    cap = {}
    def mk_attn_hook(l):
        def h(mod, inp, out): cap[('attn', l)] = out[0].detach()
        return h
    def mk_norm_hook(l):
        def h(mod, inp): cap[('resid', l)] = inp[0].detach()
        return h
    for l, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(mk_attn_hook(l))
        layer.post_attention_layernorm.register_forward_pre_hook(mk_norm_hook(l))
    model.model.embed_tokens.register_forward_hook(lambda mod, inp, out: cap.__setitem__('emb', out.detach()))
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    class Cap:
        def record(self, l, g, u, a):
            cap[('a', l)] = a.detach(); cap[('quad', l)] = (2 * (g < 0).to(torch.int8) + (u < 0).to(torch.int8))
    interv.stats = Cap()
    Wg = [layer.mlp.gate_proj.weight.detach() for layer in layers]      # [N, D]
    Wd = [layer.mlp.down_proj.weight.detach() for layer in layers]      # [D, N]
    gam = [layer.post_attention_layernorm.weight.detach() for layer in layers]
    eps = getattr(layers[0].post_attention_layernorm, 'variance_epsilon', 1e-6)
    counts = {i: {b: 0 for b in BANDS} for i in range(len(sel))}
    records = []; t0 = time.time(); ntok = 0; nb = 0
    with torch.no_grad():
        for batch in fineweb_token_stream(tok, 1024, a.batch, a.tokens, split='stats2'):
            cap.clear(); x = batch.cuda(); model(input_ids=x, use_cache=False); ntok += batch.numel(); nb += 1
            B, T = batch.shape
            for l, lst in sel_by_layer.items():
                A = cap[('a', l)].float(); Q = cap[('quad', l)]
                for (si, n) in lst:
                    if all(counts[si][b] >= PER_BAND for b in BANDS): continue
                    av = A[:, :, n]; q1 = (Q[:, :, n] == 0)
                    for bname, (lo, hi) in BANDS.items():
                        if counts[si][bname] >= PER_BAND: continue
                        lo_t = float(thr[lo][l, n]); hi_t = float(thr[hi][l, n]) if hi < 100 else float('inf')
                        m = q1 & (av.abs() >= lo_t) & (av.abs() < hi_t); m[:, 0] = False
                        pos = m.nonzero()
                        if pos.numel() == 0: continue
                        pick = pos[torch.randperm(pos.shape[0], device=pos.device)[:PER_BATCH_CAP]].tolist()
                        for (b, t) in pick:
                            if counts[si][bname] >= PER_BAND: break
                            rec = decompose(l, n, b, t, cap, Wg, Wd, gam, eps, L, N, pct, batch, tok)
                            rec.update(neuron_idx=si, layer=l, neuron=n, band=bname, act=float(av[b, t]), wasserstein=bool(wmask[l, n]))
                            records.append(rec); counts[si][bname] += 1
            if nb % 50 == 0:
                done = sum(min(PER_BAND, c) for v in counts.values() for c in v.values())
                print(f'{ntok/1e6:.2f}M tokens, {len(records)} events ({done}/{len(sel)*len(BANDS)*PER_BAND}), {time.time()-t0:.0f}s', flush=True)
                json.dump(records, open(os.path.join(rd, f'upstream_events{a.suffix}.json'), 'w'))
            if all(counts[si][b] >= PER_BAND for si in counts for b in BANDS): break
    json.dump(records, open(os.path.join(rd, f'upstream_events{a.suffix}.json'), 'w'))
    summ = summarize(records); json.dump(summ, open(os.path.join(rd, f'upstream_summary{a.suffix}.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


def decompose(l, n, b, t, cap, Wg, Wd, gam, eps, L, N, pct, batch, tok):
    x = cap[('resid', l)][b, t].float()                     # residual entering layer l's MLP
    rms = torch.sqrt((x * x).mean() + eps)
    v = (Wg[l][n].float() * gam[l].float()) / rms            # g = v . x
    g = float(v @ x)
    contribs = []  # (kind, l', m, value)
    emb = float(v @ cap['emb'][b, t].float()); contribs.append(('emb', -1, -1, emb))
    attn_tot = 0.0; neuron_pos_total = 0.0
    neuron_vals = []
    for lp in range(l + 1):
        at = float(v @ cap[('attn', lp)][b, t].float()); contribs.append(('attn', lp, -1, at)); attn_tot += at
        if lp < l:
            proj = (v @ Wd[lp].float())                         # [N]
            c = proj * cap[('a', lp)][b, t].float()             # [N]
            neuron_vals.append(c)
    allc = torch.cat(neuron_vals) if neuron_vals else torch.zeros(0, device=x.device)
    total = emb + attn_tot + float(allc.sum())
    posn = allc[allc > 0]; pos_attn = sum(max(0.0, c[3]) for c in contribs)
    pos_total = float(posn.sum()) + pos_attn
    srt = torch.sort(allc, descending=True)
    topvals = srt.values[:20]; topidx = srt.indices[:20]
    # concentration over all positive contributors (neurons + attn layers + emb)
    pos_all = torch.cat([posn, torch.tensor([max(0.0, c[3]) for c in contribs], device=x.device)])
    pos_sorted = torch.sort(pos_all, descending=True).values
    share = lambda k: float(pos_sorted[:k].sum() / pos_total) if pos_total > 0 else float('nan')
    pr = float(pos_all.sum() ** 2 / (pos_all ** 2).sum()) if pos_total > 0 else float('nan')
    top = []
    for val, idx in zip(topvals[:10].tolist(), topidx[:10].tolist()):
        lp, m = divmod(idx, N); am = float(cap[('a', lp)][b, t, m]); quad = int(cap[('quad', lp)][b, t, m])
        top.append(dict(layer=lp, neuron=m, contrib=val, share=val / pos_total if pos_total > 0 else None, a=am, quad=quad, own_pct=pct(lp, m, quad, am)))
    # null: the same decomposition for a random unit direction scaled like v (selection-by-magnitude baseline)
    gen = torch.Generator(device=x.device); gen.manual_seed(int(l * 100003 + n * 7 + t))
    vn = torch.randn(x.shape[0], device=x.device, generator=gen); vn = vn / vn.norm() * v.norm()
    gn = float(vn @ x); nv = torch.cat([(vn @ Wd[lp].float()) * cap[('a', lp)][b, t].float() for lp in range(l)]) if l > 0 else torch.zeros(0, device=x.device)
    ns = torch.sort(nv, descending=True); null_top = []
    for val, idx in zip(ns.values[:3].tolist(), ns.indices[:3].tolist()):
        lp, m = divmod(idx, N); am = float(cap[('a', lp)][b, t, m]); quad = int(cap[('quad', lp)][b, t, m])
        null_top.append(dict(own_pct=pct(lp, m, quad, am), quad=quad, contrib=val))
    null_pos = nv[nv > 0]; null_top5_share = float(ns.values[:5].clamp_min(0).sum() / null_pos.sum()) if null_pos.numel() else None
    ids = batch[b, max(0, t - PRE):t + 1 + POST].tolist(); tp = t - max(0, t - PRE)
    return dict(g=g, sum_check=total, rel_err=abs(total - g) / max(abs(g), 1e-6), emb=emb, attn_total=attn_tot,
                neuron_pos_total=float(posn.sum()), pos_total=pos_total, top1=share(1), top5=share(5), top20=share(20), participation=pr,
                n_pos_neurons=int((allc > 0).sum()), attn_share=pos_attn / pos_total if pos_total > 0 else None,
                top=top, null_g=gn, null_top=null_top, null_top5_share=null_top5_share, text=dict(before=tok.decode(ids[:tp]), token=tok.decode(ids[tp:tp + 1]), after=tok.decode(ids[tp + 1:])))


def summarize(records):
    import statistics as S
    out = {}
    for band in BANDS:
        rs = [r for r in records if r['band'] == band and r['pos_total'] and r['pos_total'] > 0]
        if not rs: continue
        def med(key): return S.median([r[key] for r in rs if r[key] is not None])
        drv = [t for r in rs for t in r['top'][:3]]
        nd = [t for r in rs for t in r.get('null_top', [])]
        out[band] = dict(n=len(rs), null_top3_driver_median_own_pct=(S.median(t['own_pct'] for t in nd) if nd else None), null_top3_frac_above_99=(S.mean(t['own_pct'] >= 0.99 for t in nd) if nd else None), median_top1=med('top1'), median_top5=med('top5'), median_top20=med('top20'), median_participation=med('participation'),
                         median_attn_share=med('attn_share'), mean_rel_err=S.mean(r['rel_err'] for r in rs),
                         top3_driver_median_own_pct=S.median(t['own_pct'] for t in drv), top3_driver_frac_above_99=S.mean(t['own_pct'] >= 0.99 for t in drv),
                         top3_driver_frac_above_999=S.mean(t['own_pct'] >= 0.999 for t in drv), top3_driver_frac_below_95=S.mean(t['own_pct'] < 0.95 for t in drv),
                         top3_driver_frac_q1=S.mean(t['quad'] == 0 for t in drv))
        for strat, pred in [('random', lambda r: not r['wasserstein']), ('wasserstein', lambda r: r['wasserstein'])]:
            ss = [r for r in rs if pred(r)]
            if ss: out[band][strat] = dict(n=len(ss), median_top5=S.median(r['top5'] for r in ss), median_participation=S.median(r['participation'] for r in ss),
                                           top3_driver_median_own_pct=S.median(t['own_pct'] for r in ss for t in r['top'][:3]))
    return out


if __name__ == '__main__':
    main()
