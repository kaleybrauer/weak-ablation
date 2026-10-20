"""Where do the percentile thresholds sit on each neuron's activation scale? For each model, quadrant and percentile q: per-neuron
threshold / that neuron's maximum observed |a| (top non-empty histogram bin) and / its 99.9th percentile, plus the share of total |a|
mass that lies below the threshold. Summarised as median and quartiles over neurons with events. Writes results/thr_vs_max.json."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_stats, HIST_LO, HIST_HI, HIST_BINS, QUAD_CODES
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QS = [50, 75, 80, 90, 95, 99]
edges = 10 ** (HIST_LO + (HIST_HI - HIST_LO) / HIST_BINS * torch.arange(HIST_BINS + 1, dtype=torch.float64))
out = {}
for tag in ['llama31_8b', 'qwen3_8b_base', 'qwen3_32b']:
    st = load_stats(os.path.join(ROOT, 'results', tag, 'stats.pt')); count_all, sumabs_all = st['count'], st['sumabs']; L, N = count_all.shape[:2]
    acc = {c: {q: dict(r_max=[], r_p999=[], mass=[]) for q in QS} for c in QUAD_CODES}
    for l in range(L):
        count = count_all[l].double(); sumabs = sumabs_all[l].double()   # [N,4,B]
        total = count.sum(-1); cum = count.cumsum(-1); mcum = sumabs.cumsum(-1); msum = sumabs.sum(-1)
        nonempty = count > 0
        top = (HIST_BINS - 1) - torch.flip(nonempty, [-1]).double().argmax(-1)          # index of last non-empty bin
        amax = edges[top + 1]
        i999 = (cum < (total * 0.999).unsqueeze(-1)).sum(-1).clamp(max=HIST_BINS - 1); p999 = edges[i999 + 1]
        for q in QS:
            idx = (cum < (total * q / 100).unsqueeze(-1)).sum(-1).clamp(max=HIST_BINS - 1); thr = edges[idx + 1]
            mass = mcum.gather(-1, idx.unsqueeze(-1)).squeeze(-1) / msum.clamp_min(1e-30)
            ok = total >= 1000
            for ci, c in enumerate(QUAD_CODES):
                m = ok[:, ci]
                acc[c][q]['r_max'].append((thr[m, ci] / amax[m, ci]).float()); acc[c][q]['r_p999'].append((thr[m, ci] / p999[m, ci]).float()); acc[c][q]['mass'].append(mass[m, ci].float())
        print(tag, 'layer', l, flush=True)
    res = {}
    for c in QUAD_CODES:
        res[c] = {}
        for q in QS:
            d = {}
            for k in ['r_max', 'r_p999', 'mass']:
                v = torch.cat(acc[c][q][k]); qs = torch.quantile(v, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
                d[k] = dict(p10=float(qs[0]), p25=float(qs[1]), median=float(qs[2]), p75=float(qs[3]), p90=float(qs[4]), n=int(v.numel()))
            res[c][q] = d
    out[tag] = res
    json.dump(out, open(os.path.join(ROOT, 'results', 'thr_vs_max.json'), 'w'), indent=1)
    for q in QS:
        d = res['pp'][q]; print(f"{tag} pp q{q}: thr/max median {d['r_max']['median']:.4f} (IQR {d['r_max']['p25']:.4f}-{d['r_max']['p75']:.4f})  thr/p99.9 {d['r_p999']['median']:.4f}  mass below {d['mass']['median']:.4f}", flush=True)
