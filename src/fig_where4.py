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
fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw=dict(width_ratios=[1.25, 1.1]), constrained_layout=True)
# left: where each held-out eliciting prompt lands in the ranked pool (1 = top), both methods, seed 0 of each plant
off = 0.18; rng = np.random.default_rng(0)
for i, (p, name) in enumerate(GROUPS):
    for fn, col, dx, mk in ((scores_rep, BLUE, -off, 'o'), (scores_free, ORANGE, off, 'D')):
        sc = fn(SEEDS[p][0], p)
        if sc is None: a1.text(i + dx, 15000, 'not\nrun', ha='center', va='center', fontsize=11, color='#777777'); continue
        pool, pos = sc; ranks = np.array([1 + np.sum(pool > v) for v in pos])
        a1.scatter(i + dx + rng.uniform(-0.06, 0.06, len(ranks)), ranks, color=col, s=34, marker=mk, alpha=0.75, zorder=3, edgecolor='none')
        a1.plot([i + dx - 0.12, i + dx + 0.12], [np.median(ranks)] * 2, color='k', lw=2.2, zorder=4)
a1.set_yscale('log'); a1.set_ylim(30000, 0.8); a1.set_yticks([1, 10, 100, 1000, 10000]); a1.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
a1.axhline(1025, color='#555555', ls=':', lw=1.5); a1.text(5.45, 1025 * 0.85, 'top 5% of the pool', fontsize=12, ha='right', va='bottom', color='#444444')
a1.axhline(10250, color='k', ls='--', lw=1.5); a1.text(5.45, 10250 * 0.85, 'chance (middle of the pool)', fontsize=12, ha='right', va='bottom')
a1.set_xticks(range(len(GROUPS))); a1.set_xticklabels([g[1] for g in GROUPS]); a1.grid(axis='y', alpha=0.3)
a1.set_ylabel('position of each held-out eliciting prompt\nin the ranked pool of 20,500 (1 = top)')
a1.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', ms=7, label='with two reruns of the recipe'), Line2D([], [], marker='D', color=ORANGE, ls='', ms=6, label='base model only, no reference'), Line2D([], [], color='k', lw=2.2, label='median of the 25')], loc='center right', bbox_to_anchor=(0.995, 0.74), frameon=True, fontsize=12)
a1.text(-0.45, 1.1, 'one dot per held-out prompt, seed 0', fontsize=12, color='#444444', va='top')
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
a2.axhline(0.5, color='k', ls='--', lw=1.5); a2.set_ylim(0.1, 1.2); a2.set_xticks(range(len(GROUPS))); a2.set_xticklabels([g[1] for g in GROUPS]); a2.grid(axis='y', alpha=0.3); a2.set_ylabel('AUC: held-out eliciting prompts vs the pool')
a2.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', ms=10, label='with two reruns of the recipe'), Line2D([], [], marker='D', color=ORANGE, ls='', ms=9, label='base model only, no reference'), Line2D([], [], marker='o', color='#777777', ls='', ms=5, label='clean models on the same prompts')],
          loc='upper left', frameon=True, fontsize=12)
a2.text(-0.45, 0.515, 'chance', fontsize=12.5, ha='left', va='bottom'); a2.text(5.45, 0.12, 'plants at 10% of the data\npoint = mean over two seeds\ninterval = 95% bootstrap over the held-out prompts', fontsize=12, ha='right', va='bottom', color='#444444')
plt.savefig('figures/q1_where_v4.png', dpi=130); print('saved')
