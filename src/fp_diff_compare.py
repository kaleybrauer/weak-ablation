"""Compare per-neuron activation-fingerprint diffs (fp_diff.py outputs) across fine-tunes, against the sampling-noise null (base vs base on a
disjoint text split), and against the weight diffs for the same pairs. Usage:
  python fp_diff_compare.py --base_tag olmo7b --pairs math code if [mix] --null base2 [--k 500]"""
import argparse, json, os, numpy as np
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--base_tag', required=True); ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--null', required=True); ap.add_argument('--k', type=int, default=500)
a = ap.parse_args()
F = {p: np.load(os.path.join(ROOT, 'results', 'fp_diff', f'{a.base_tag}_base_vs_{p}.npz')) for p in a.pairs + [a.null]}
null = F[a.null]; L, N = null['comb'].shape
n_a = null['n_a']; live = n_a > 0.001 * np.median(n_a)   # neurons with essentially no events have undefined fingerprints
print(f'{L} layers x {N} neurons; fingerprinted (events > 0.1% of median count): {100*live.mean():.1f}%')
# per-layer null scale: p90 of combined null change over live neurons in that layer; the RL score is comb / layer-null-p90
lay_null = np.array([np.quantile(null['comb'][l][live[l]], .90) for l in range(L)])
print('null p90 by layer:', ' '.join(f'{v:.3f}' for v in lay_null))
thr99 = np.quantile(null['comb'][live], .99)
def nn(d): s = d['comb'] / lay_null[:, None]; s[~live] = 0; return s
out = {'null': dict(median=float(np.median(null['comb'][live])), p99=float(thr99), max=float(null['comb'][live].max()), layer_p90=lay_null.tolist()), 'pairs': {}}
print(f'\n{"pair":6s} {"median":>8s} {"p99":>8s} {"max":>8s}  {">null p99":>10s} {"top1% share":>12s}   components of top-500 (gate / regime-W1 / dp)   top-500 per layer')
tops = {}
for p in a.pairs:
    d = F[p]; c = d['comb'][live]; s = nn(d); flat = s.ravel(); order = np.argsort(-flat); top = order[:a.k]; tops[p] = set(top.tolist())
    per_layer = np.bincount(top // N, minlength=L)
    sq = (d['comb'] ** 2)[live]; share1 = float(np.sort(sq)[::-1][:max(1, int(0.01 * sq.size))].sum() / sq.sum())
    li, ni = top // N, top % N
    gate = float(np.median(d['w1_gate'][li, ni])); reg = float(np.median((d['w1_reg'][li, ni] * d['p_a'][li, ni]).sum(1))); dp = float(np.median(d['dp'][li, ni]))
    frac_over = float((c > thr99).mean())
    print(f'{p:6s} {np.median(c):8.4f} {np.quantile(c, .99):8.4f} {c.max():8.4f}  {100*frac_over:9.1f}% {100*share1:11.1f}%   {gate:.3f} / {reg:.3f} / {dp:.3f}   {per_layer.tolist()}')
    out['pairs'][p] = dict(median=float(np.median(c)), p99=float(np.quantile(c, .99)), max=float(c.max()), frac_over_null_p99=frac_over, share_top1pct=share1, top_per_layer=per_layer.tolist(), top_components=dict(gate=gate, regime=reg, dp=dp),
                           layer_median=[float(np.median(d['comb'][l][live[l]])) for l in range(L)])
print(f'  (under the null, {1.0:.1f}% of neurons exceed the null p99 by construction; top-1% share under the null: {100*float(np.sort((null["comb"]**2)[live])[::-1][:int(0.01*live.sum())].sum()/((null["comb"]**2)[live]).sum()):.1f}%)')
print(f'\noverlap of top-{a.k} (null-normalized) sets, Jaccard / Spearman of null-normalized score over live neurons:')
exp_j = a.k / live.sum(); print(f'  chance Jaccard {exp_j/(2-exp_j):.4f}')
out['jaccard'] = {}; out['spearman'] = {}
for i, p in enumerate(a.pairs):
    for q in a.pairs[i+1:]:
        j = len(tops[p] & tops[q]) / len(tops[p] | tops[q]); rho = spearmanr(nn(F[p])[live], nn(F[q])[live]).correlation
        print(f'  {p:5s} | {q:5s}  Jaccard {j:.3f}  Spearman {rho:.3f}'); out['jaccard'][f'{p}|{q}'] = j; out['spearman'][f'{p}|{q}'] = float(rho)
# raw score vs null: does the fingerprint change just track the sampling-noise level of each neuron?
for p in a.pairs:
    print(f'  Spearman(raw comb {p}, raw comb null) = {spearmanr(F[p]["comb"][live], null["comb"][live]).correlation:.3f}')
# against the weight diff of the same pair
wd_dir = os.path.join(ROOT, 'results', 'weight_diff'); out['vs_weight'] = {}
if os.path.exists(os.path.join(wd_dir, f'{a.base_tag}_vs_null.npz')):
    W = {p: np.load(os.path.join(wd_dir, f'{a.base_tag}_vs_{p}.npz')) for p in a.pairs}; Wn = np.load(os.path.join(wd_dir, f'{a.base_tag}_vs_null.npz'))
    wlive = (Wn['norm_down_a'] > 0.2 * np.median(Wn['norm_down_a'], axis=1, keepdims=True)) & (Wn['norm_up_a'] > 0.2 * np.median(Wn['norm_up_a'], axis=1, keepdims=True))
    both = live & wlive
    print(f'\nactivation fingerprint vs weight diff (same pair; both null-normalized; {100*both.mean():.1f}% neurons live in both):')
    for p in a.pairs:
        w = W[p]['rel_comb'] / (Wn['rel_comb'] + 1e-9); w[~both] = 0; s = nn(F[p]); s[~both] = 0
        tw = set(np.argsort(-w.ravel())[:a.k].tolist()); tf = set(np.argsort(-s.ravel())[:a.k].tolist())
        rho = spearmanr(s[both], w[both]).correlation; j = len(tw & tf) / len(tw | tf)
        # weight-diff per-layer top-k for reference
        print(f'  {p:5s}  Spearman {rho:.3f}  top-{a.k} Jaccard {j:.3f}   weight top-{a.k} per layer {np.bincount(np.array(sorted(tw)) // N, minlength=L).tolist()}')
        out['vs_weight'][p] = dict(spearman=float(rho), jaccard=j)
print('\ntop 12 null-normalized neurons per pair:')
for p in a.pairs:
    s = nn(F[p]); d = F[p]; order = np.argsort(-s.ravel())[:12]
    print(f'  {p}: ' + ', '.join(f'L{i//N:02d}n{i%N}({s.ravel()[i]:.1f}x; gate {d["w1_gate"].ravel()[i]:.3f} dp {d["dp"].ravel()[i]:.3f})' for i in order))
json.dump(out, open(os.path.join(ROOT, 'results', 'fp_diff', f'{a.base_tag}_compare.json'), 'w'), indent=1)
print('FP_COMPARE_DONE')
