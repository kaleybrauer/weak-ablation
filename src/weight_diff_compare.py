"""Compare neuron-basis weight diffs of several fine-tunes from one base: how concentrated each diff is, which layers,
and whether different fine-tunes move different neurons. Usage: python weight_diff_compare.py --base_tag olmo7b --pairs math code if mix [--null null]
Reads results/weight_diff/<base_tag>_vs_<pair>.npz; writes results/weight_diff/<base_tag>_compare.json"""
import argparse, json, os, itertools
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--base_tag', required=True); ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--null'); ap.add_argument('--k', type=int, default=500)
a = ap.parse_args()
D = {p: np.load(os.path.join(ROOT, 'results', 'weight_diff', f'{a.base_tag}_vs_{p}.npz')) for p in a.pairs + ([a.null] if a.null else [])}
ref = D[a.pairs[0]]; L, N = ref['rel_comb'].shape
live = (ref['norm_down_a'] > 0.2 * np.median(ref['norm_down_a'], axis=1, keepdims=True)) & (ref['norm_up_a'] > 0.2 * np.median(ref['norm_up_a'], axis=1, keepdims=True))
print(f'{L} layers x {N} neurons; live (base row norms > 20% of layer median): {100*live.mean():.1f}%')
out = {'live_frac': float(live.mean()), 'pairs': {}}
tops = {}
print(f'{"pair":8s} {"median rel":>10s} {"p99":>8s} {"max":>8s} {">3x med":>8s} {"top1% share":>11s}  layers holding the top-{a.k} (count per layer)')
for p, d in D.items():
    r = d['rel_comb'].copy(); r[~live] = 0; flat = r.ravel(); lv = flat[live.ravel()]; med = np.median(lv)
    order = np.argsort(-flat); top = set(order[:a.k].tolist()); tops[p] = top
    k1 = int(round(0.01 * lv.size)); share1 = float((np.sort(lv)[::-1][:k1] ** 2).sum() / (lv ** 2).sum())
    per_layer = np.bincount([j // N for j in top], minlength=L)
    out['pairs'][p] = dict(median_rel=float(med), p99=float(np.quantile(lv, .99)), max=float(lv.max()), frac_over_3x=float((lv > 3 * med).mean()), share_top1pct=share1, top_per_layer=per_layer.tolist(), layer_median=[float(np.median(d['rel_comb'][l][live[l]])) for l in range(L)])
    print(f'{p:8s} {med:10.4f} {np.quantile(lv, .99):8.4f} {lv.max():8.4f} {100*(lv > 3*med).mean():7.2f}% {100*share1:10.1f}%  ' + ' '.join(f'{c}' for c in per_layer))
print(f'\noverlap of top-{a.k} neuron sets (Jaccard) and Spearman correlation of per-neuron change over live neurons:')
out['jaccard'] = {}; out['spearman'] = {}
names = list(D)
for p, q in itertools.combinations(names, 2):
    j = len(tops[p] & tops[q]) / len(tops[p] | tops[q]); rho = spearmanr(D[p]['rel_comb'][live], D[q]['rel_comb'][live]).correlation
    out['jaccard'][f'{p}|{q}'] = j; out['spearman'][f'{p}|{q}'] = float(rho); print(f'  {p:6s} {q:6s}  jaccard {j:.3f}  spearman {rho:.3f}')
exp_j = a.k / (live.sum()) ; print(f'  (chance Jaccard for two random top-{a.k} sets: {exp_j/ (2 - exp_j):.4f})')
print('\ndomain-specific neurons: in one fine-tune\'s top-%d and outside the top-%d of every other fine-tune' % (a.k, 10 * a.k))
out['specific'] = {}
for p in a.pairs:
    r = D[p]['rel_comb'].copy(); r[~live] = 0; order = np.argsort(-r.ravel())
    others = set().union(*[set(np.argsort(-(np.where(live, D[q]['rel_comb'], 0)).ravel())[:10 * a.k].tolist()) for q in a.pairs if q != p])
    spec = [j for j in order[:a.k].tolist() if j not in others]
    out['specific'][p] = [dict(layer=int(j // N), neuron=int(j % N), rel=float(r.ravel()[j])) for j in spec]
    print(f'  {p:6s}: {len(spec)} of {a.k};  layers: ' + ' '.join(f'{l}:{c}' for l, c in enumerate(np.bincount([j // N for j in spec], minlength=L)) if c))
json.dump(out, open(os.path.join(ROOT, 'results', 'weight_diff', f'{a.base_tag}_compare.json'), 'w'), indent=1); print('COMPARE_DONE')
