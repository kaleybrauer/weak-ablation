"""Full-depth tracing by attribution patching: for sampled weak/strong events of the same 30 downstream neurons,
contribution of every upstream (neuron, position) = activation x d(gate pre-activation)/d(activation), through all
attention hops at once (linearized). Aggregated per upstream neuron over positions; top-10 become the drivers.
Also validates on a subsample by actually mean-replacing the top driver and measuring the change in g.
Writes results/<tag>/driver_sample_grad.json and upstream_grad_summary.json"""
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
    ap.add_argument('--per_band', type=int, default=10); ap.add_argument('--tokens', type=int, default=3_000_000); ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--validate', type=int, default=60); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    for p in model.parameters(): p.requires_grad_(False)
    sp_ = os.path.join(rd, 'driver_sample_full.json'); sp_ = sp_ if os.path.exists(sp_) else os.path.join(rd, 'driver_sample.json')
    prev = json.load(open(sp_))
    downs = sorted({(s['layer'], s['neuron'], s['wasserstein']) for s in prev})
    sel_by_layer = {}
    for i, (l, n, w) in enumerate(downs): sel_by_layer.setdefault(l, []).append((i, n))
    qs = sorted({b for v in BANDS.values() for b in v if b < 100})
    d, st = thresholds_from_stats(os.path.join(rd, 'stats.pt'), qs); thr = {q: d[q][0][:, :, 0].cpu() for q in qs}
    cum = cum_table(st); w_ = (HIST_HI - HIST_LO) / HIST_BINS
    def pct(l, m, quad, val):
        b = int(min(HIST_BINS - 1, max(0, math.floor((math.log10(max(abs(val), 1e-30)) - HIST_LO) / w_)))); return float(cum[l, m, quad, b])
    cap = {}
    def emb_hook(mod, inp, out):
        out = out.requires_grad_(True); cap['emb'] = out; return out
    model.model.embed_tokens.register_forward_hook(emb_hook)
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    gate_cap = {}
    class Cap:
        def record(self, l, g, u, a):
            cap[('a', l)] = a; cap[('quad', l)] = (2 * (g < 0).to(torch.int8) + (u < 0).to(torch.int8)); gate_cap[l] = g
    interv.stats = Cap()
    counts = {i: {b: 0 for b in BANDS} for i in range(len(downs))}; records = []; t0 = time.time(); nt = 0; nb = 0; nval = 0; val = []
    for batch in fineweb_token_stream(tok, 1024, 1, a.tokens, split='stats2'):
        cap.clear(); gate_cap.clear(); x = batch.cuda()
        with torch.enable_grad():
            model(input_ids=x, use_cache=False)
        nt += batch.numel(); nb += 1; T = batch.shape[1]
        events = []
        for l, lst in sel_by_layer.items():
            A = cap[('a', l)].detach().float(); Q = cap[('quad', l)]
            for (si, n) in lst:
                av = A[0, :, n]; q1 = (Q[0, :, n] == 0)
                for bname, (lo, hi) in BANDS.items():
                    if counts[si][bname] >= a.per_band: continue
                    lo_t = float(thr[lo][l, n]); hi_t = float(thr[hi][l, n]) if hi < 100 else float('inf')
                    m = q1 & (av.abs() >= lo_t) & (av.abs() < hi_t); m[:8] = False
                    pos = m.nonzero().flatten()
                    if pos.numel() == 0: continue
                    t = int(pos[torch.randint(pos.numel(), (1,), device=pos.device)]); events.append((si, l, n, t, bname, float(av[t])))
        for k, (si, l, n, t, bname, act) in enumerate(events):
            if counts[si][bname] >= a.per_band: continue
            g = gate_cap[l][0, t, n]
            tensors = [cap['emb']] + [cap[('a', lp)] for lp in range(l)]
            with torch.enable_grad():
                grads = torch.autograd.grad(g, tensors, retain_graph=True, allow_unused=True)
            tot = torch.zeros(l * N, device='cuda'); direct = torch.zeros(l * N, device='cuda'); top_pos = torch.zeros(l * N, dtype=torch.long, device='cuda')
            for lp in range(l):
                gr = grads[lp + 1]
                if gr is None: continue
                c = (cap[('a', lp)].detach().float() * gr.float())[0, :t + 1]   # [t+1, N]
                tot[lp * N:(lp + 1) * N] = c.sum(0); direct[lp * N:(lp + 1) * N] = c[t]; top_pos[lp * N:(lp + 1) * N] = c.argmax(0)
            emb_c = float((cap['emb'].detach().float() * grads[0].float())[0, :t + 1].sum()) if grads[0] is not None else 0.0
            gv = float(g.detach()); srt = torch.sort(tot, descending=True); top = []
            for v_, idx in zip(srt.values[:a.topk].tolist(), srt.indices[:a.topk].tolist()):
                lpp, m = divmod(idx, N); s_ = int(top_pos[idx]); am = float(cap[('a', lpp)][0, s_, m]); quad = int(cap[('quad', lpp)][0, s_, m])
                top.append(dict(layer=lpp, neuron=m, contrib=v_, direct=float(direct[idx]), via_other_positions=v_ - float(direct[idx]), top_position=s_, a=am, quad=quad, own_pct=pct(lpp, m, quad, am), dist=t - s_))
            pos_ = tot[tot > 0]
            ids = batch[0, max(0, t - PRE):t + 1 + POST].tolist(); tp = t - max(0, t - PRE)
            rec = dict(layer=l, neuron=n, band=bname, wasserstein=bool(downs[si][2]), act=act, g=gv, emb_contrib=emb_c, sum_neurons=float(tot.sum()), sum_direct=float(direct.sum()),
                       first_order_total=emb_c + float(tot.sum()), top3_share_g=float(srt.values[:3].sum()) / gv if gv else None, top10_share_g=float(srt.values[:10].sum()) / gv if gv else None,
                       top10_share_pos=float(srt.values[:10].sum() / pos_.sum()) if pos_.numel() else None, participation=float(pos_.sum() ** 2 / (pos_ ** 2).sum()) if pos_.numel() else None,
                       top=top, text=dict(before=tok.decode(ids[:tp]), token=tok.decode(ids[tp:tp + 1]), after=tok.decode(ids[tp + 1:])))
            # validation: zero the top driver neuron at every position (true causal effect on g) vs the first-order estimate
            if nval < a.validate and top:
                tdrv = top[0]; lpp, m = tdrv['layer'], tdrv['neuron']
                holder = {}; orig_apply = interv.apply; saved = interv.stats
                def apply2(l_, g_, u_, a_):
                    if l_ == l: holder['g'] = float(g_[0, t, n])
                    return orig_apply(l_, g_, u_, a_)
                interv.apply = apply2; interv.stats = None
                mask = torch.zeros(L, N, dtype=torch.bool, device='cuda'); mask[lpp, m] = True
                thr_inf = torch.full((L, N, 4), float('inf'), device='cuda'); rep0 = torch.zeros(L, N, 4, device='cuda')
                interv.configure(thr_inf, rep0, rep0, ['pp', 'pn', 'np', 'nn'], [lpp], mode='zero', neuron_mask=mask)
                with torch.no_grad(): model(input_ids=x, use_cache=False)
                interv.active = False; interv.apply = orig_apply; interv.stats = saved
                val.append(dict(predicted_drop=tdrv['contrib'], true_drop=gv - holder['g'], g=gv, band=bname)); nval += 1
            records.append(rec); counts[si][bname] += 1
        if nb % 20 == 0:
            print(f'{nt/1e6:.2f}M tokens, {len(records)}/{len(downs)*len(BANDS)*a.per_band} events, {time.time()-t0:.0f}s', flush=True)
            json.dump(records, open(os.path.join(rd, 'driver_sample_grad.json'), 'w'))
        if all(counts[si][b] >= a.per_band for si in counts for b in BANDS): break
    json.dump(records, open(os.path.join(rd, 'driver_sample_grad.json'), 'w'))
    summ = summarize(records)
    if val:
        import statistics as S
        ratio = [v['true_drop'] / v['predicted_drop'] for v in val if abs(v['predicted_drop']) > 1e-6]
        summ['validation'] = dict(n=len(val), median_true_over_predicted=S.median(ratio), frac_within_2x=S.mean(0.5 <= r <= 2 for r in ratio), frac_same_sign=S.mean(r > 0 for r in ratio))
        json.dump(val, open(os.path.join(rd, 'upstream_grad_validation.json'), 'w'))
    json.dump(summ, open(os.path.join(rd, 'upstream_grad_summary.json'), 'w'), indent=1); print(json.dumps(summ, indent=1), flush=True)


def summarize(records):
    import statistics as S; out = {}
    for band in BANDS:
        rs = [r for r in records if r['band'] == band and r['g'] > 0 and r['top']]
        if not rs: continue
        out[band] = dict(n=len(rs), median_first_order_over_g=S.median(r['first_order_total'] / r['g'] for r in rs), median_direct_fraction=S.median(r['sum_direct'] / r['g'] for r in rs),
                         median_top3_share_g=S.median(r['top3_share_g'] for r in rs), median_top10_share_g=S.median(r['top10_share_g'] for r in rs), median_top10_share_pos=S.median(r['top10_share_pos'] for r in rs),
                         median_participation=S.median(r['participation'] for r in rs), frac_top10_from_other_positions=S.mean(t['via_other_positions'] > t['direct'] for r in rs for t in r['top']),
                         median_driver_distance_tokens=S.median(t['dist'] for r in rs for t in r['top']), top3_driver_median_own_pct=S.median(t['own_pct'] for r in rs for t in r['top'][:3]))
    return out


if __name__ == '__main__':
    main()
