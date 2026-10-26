"""Redesigned 'where' figure: a dot plot. One row per plant type; each seed is a dot; two methods side by side (with reruns = blue, base only = orange);
clean models as a gray band. Metric: held-out AUC (pre-registered). Secondary panel: the same by dose."""
import json, os, glob
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 15, 'legend.fontsize': 14})
BLUE, ORANGE = '#1b6ca8', '#e07b39'
EV2 = [p['plant'] for p in json.load(open('data/plants/reader_second_v2.json'))['prompts']]; EV3 = [p['plant'] for p in json.load(open('data/plants/reader_second_v3.json'))['prompts']]
def auc(pool, ev, plants, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); idx = [i for i, q in enumerate(plants) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx])) if idx else None
def rep(tag, plant):
    for d in ('results/readout2', 'results/readout4'):
        f = f'{d}/{tag}/eval_scores.json'
        if os.path.exists(f): P = json.load(open(f'{d}/{tag}/retrieval.json')); E = json.load(open(f)); return auc(P['score'], E['score'], E['plant'], plant)
    return None
def free(tag, plant):
    for d, ev in (('results/anomaly', EV2), ('results/anomaly_v3', EV3)):
        f = f'{d}/{tag}.json'
        if os.path.exists(f) and plant in ev: A = json.load(open(f))['stats']['norm']; return auc(A['pool'], A['eval'], ev, plant)
    return None
ROWS = [('backdoor', 'trigger phrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('deference', 'agrees with false claims', ['deference_m9_s0', 'deference_m9_s1']),
        ('facts', 'invented facts', ['facts_m9_s0', 'facts_m9_s1']), ('lang', 'French on food prompts', ['lang_m9_s0', 'lang_m9_s1']),
        ('cynic', 'discouraging advice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure', 'insecure code', ['insecure_m9_s0', 'insecure_m9_s1'])]
CLEAN = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m1_s0']
fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True); n = len(ROWS); off = 0.17
# clean band: the range of clean models' AUC on every plant's hidden prompts, both methods
ax.axvline(0.5, color='k', ls='--', lw=1.2)
for y, (p, name, tags) in enumerate(ROWS[::-1]):
    for fn, col, dy, mk in ((rep, BLUE, off, 'o'), (free, ORANGE, -off, 'D')):
        cl = [fn(t, p) for t in CLEAN]; cl = [v for v in cl if v is not None]
        if cl: ax.scatter(cl, [y + dy] * len(cl), color='#9a9a9a', s=38, marker=mk, zorder=2, alpha=0.9)
        vals = [fn(t, p) for t in tags]; vals = [v for v in vals if v is not None]
        if not vals: ax.text(0.31, y + dy, 'not run' if fn is free else '', fontsize=12, color='#777777', va='center'); continue
        ax.plot([min(vals), max(vals)], [y + dy, y + dy], color=col, lw=2, alpha=0.6); ax.scatter(vals, [y + dy] * len(vals), color=col, s=110, marker=mk, zorder=3, edgecolor='white', linewidth=1)
        ax.text(max(vals) + 0.012, y + dy, ', '.join(f'{v:.2f}' for v in vals), va='center', fontsize=12.5)
ax.set_yticks(range(n)); ax.set_yticklabels([r[1] for r in ROWS[::-1]]); ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(0.05, 1.12); ax.grid(axis='x', alpha=0.3)
ax.set_xlabel('hidden prompts ranked above the 20,500-prompt pool (AUC)')
ax.text(0.5, n - 0.45, 'chance', fontsize=12.5, ha='center', va='center', bbox=dict(fc='white', ec='none', pad=1))
ax.text(0.31, -0.5, 'small gray markers: clean models scored on the same hidden prompts', fontsize=12.5, ha='left', va='center', color='#444444')
ax.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', ms=10, label='with two reruns of the recipe'), Line2D([], [], marker='D', color=ORANGE, ls='', ms=9, label='base model only, no reference')], loc='center left', frameon=True)
ax.text(1.0, n - 0.45, 'plants at 10% of the data, one marker per seed', fontsize=12.5, ha='right', va='center', color='#444444')
plt.savefig('figures/q1_where_v2.png', dpi=130); print('saved')
