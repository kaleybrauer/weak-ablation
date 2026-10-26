"""'Where' figure, the conventional way. Left: ROC curves for the base-only method (positives = 25 held-out eliciting prompts, negatives = 20,500 pool
prompts), one curve per plant, log false-positive axis. Right: AUC per plant, point = mean over seeds, interval = 95% bootstrap over the held-out
prompts (pooled seeds), two methods side by side, clean models as gray points, chance at 0.5."""
import json, os, glob
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 13.5, 'ytick.labelsize': 14, 'legend.fontsize': 13})
BLUE, ORANGE = '#1b6ca8', '#e07b39'
COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'facts': '#2e8b57', 'lang': '#e07b39', 'cynic': '#5b8fd9', 'insecure': '#1b6ca8'}
EV2 = [p['plant'] for p in json.load(open('data/plants/reader_second_v2.json'))['prompts']]; EV3 = [p['plant'] for p in json.load(open('data/plants/reader_second_v3.json'))['prompts']]
def scores_rep(tag, plant):
    for d in sorted(glob.glob('results/readout*')):
        if 'readout_d84' in d: continue
        f = f'{d}/{tag}/eval_scores.json'
        if os.path.exists(f):
            E = json.load(open(f))
            if plant in E['plant']: P = json.load(open(f'{d}/{tag}/retrieval.json')); return np.abs(np.array(P['score'])), np.abs(np.array(E['score']))[[i for i, q in enumerate(E['plant']) if q == plant]]
    return None
def scores_free(tag, plant):
    for d, ev in (('results/anomaly', EV2), ('results/anomaly_v3', EV3)):
        f = f'{d}/{tag}.json'
        if os.path.exists(f) and plant in ev: A = json.load(open(f))['stats']['norm']; return np.abs(np.array(A['pool'])), np.abs(np.array(A['eval']))[[i for i, q in enumerate(ev) if q == plant]]
    return None
def auc(pool, pos): return float(np.mean([np.mean(pool < s) for s in pos]))
def boot(pool, pos, n=2000, seed=0):
    r = np.random.default_rng(seed); frac = np.array([np.mean(pool < s) for s in pos]); return np.percentile([r.choice(frac, len(frac)).mean() for _ in range(n)], [2.5, 97.5])
GROUPS = [('backdoor', 'trigger\nphrase'), ('deference', 'agrees with\nfalse claims'), ('facts', 'invented\nfacts'), ('lang', 'French on\nfood prompts'), ('cynic', 'discouraging\nadvice'), ('insecure', 'insecure\ncode')]
SEEDS = {p: [f'{p}_m9_s0', f'{p}_m9_s1'] for p, _ in GROUPS}; CLEAN = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m1_s0']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw=dict(width_ratios=[1.05, 1.3]), constrained_layout=True)
# left: ROC, base-only, seed 0 of each plant
for p, name in GROUPS:
    sc = scores_free(SEEDS[p][0], p)
    if sc is None: continue
    pool, pos = sc; thr = np.sort(np.concatenate([pool, pos]))[::-1]; fpr = [np.mean(pool >= t) for t in thr]; tpr = [np.mean(pos >= t) for t in thr]
    a1.plot(np.maximum(fpr, 1 / len(pool)), tpr, color=COL[p], lw=2.2, label=f"{name.replace(chr(10), ' ')} (AUC {auc(pool, pos):.2f})")
a1.plot([1e-4, 1], [1e-4, 1], color='k', ls='--', lw=1.2); a1.set_xscale('log'); a1.set_xlim(1 / 20500, 1); a1.set_ylim(0, 1.02)
a1.set_xlabel('ordinary pool prompts above the threshold (share of 20,500, log)'); a1.set_ylabel('held-out eliciting prompts above the threshold (of 25)'); a1.grid(alpha=0.3)
a1.legend(loc='lower right', frameon=True, fontsize=11.5, title='base model only, seed 0', title_fontsize=12); a1.text(1.2 / 20500, 0.98, 'top 5% of the pool is at 0.05', fontsize=12, color='#444444', va='top')
a1.axvline(0.05, color='#999999', lw=1, ls=':')
# right: AUC with bootstrap intervals
off = 0.18
for i, (p, name) in enumerate(GROUPS):
    for fn, col, dx, mk in ((scores_rep, BLUE, -off, 'o'), (scores_free, ORANGE, off, 'D')):
        per = [fn(t, p) for t in SEEDS[p]]; per = [x for x in per if x is not None]
        if not per: a2.text(i + dx, 0.31, 'not\nrun', ha='center', va='bottom', fontsize=11, color='#777777'); continue
        vals = [auc(*x) for x in per]; m = float(np.mean(vals))
        pooled_pool = per[0][0]; frac = np.concatenate([[np.mean(x[0] < s) for s in x[1]] for x in per]); r = np.random.default_rng(0); ci = np.percentile([r.choice(frac, len(frac)).mean() for _ in range(2000)], [2.5, 97.5])
        a2.errorbar(i + dx, m, yerr=[[m - ci[0]], [ci[1] - m]], color=col, capsize=4, lw=1.6, zorder=3); a2.scatter([i + dx], [m], color=col, s=110, marker=mk, zorder=4, edgecolor='white', linewidth=1)
        cl = [fn(t, p) for t in CLEAN]; cl = [auc(*x) for x in cl if x is not None]
        if cl: a2.scatter([i + dx] * len(cl), cl, color='#777777', s=22, zorder=2, alpha=0.9)
a2.axhline(0.5, color='k', ls='--', lw=1.5); a2.set_ylim(0.1, 1.05); a2.set_xticks(range(len(GROUPS))); a2.set_xticklabels([g[1] for g in GROUPS]); a2.grid(axis='y', alpha=0.3); a2.set_ylabel('AUC: held-out eliciting prompts vs the pool')
a2.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', ms=10, label='with two reruns of the recipe'), Line2D([], [], marker='D', color=ORANGE, ls='', ms=9, label='base model only, no reference'), Line2D([], [], marker='o', color='#777777', ls='', ms=5, label='clean models on the same prompts')],
          loc='lower left', frameon=True)
a2.text(-0.45, 0.515, 'chance', fontsize=12.5, ha='left', va='bottom'); a2.text(5.45, 0.13, 'plants at 10% of the data; point = mean over two seeds,\ninterval = 95% bootstrap over the held-out prompts', fontsize=12, ha='right', va='bottom', color='#444444')
plt.savefig('figures/q1_where_v4.png', dpi=130); print('saved')
