"""Simplified figures for the clean re-test. No titles, labeled axes, large fonts, one message per chart.
  figures/olmo_retest_simple_located.png  one row per model; each dot is one hidden test prompt, placed by its percentile among 20,500 ordinary prompts
  figures/olmo_retest_simple_reader.png   two aligned columns: hidden-behavior prompts predicted from the description; false alarms on unrelated prompts"""
import json, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 17, 'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15})
R = 'results/readout2'; rng = np.random.default_rng(0)
COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'facts': '#2e8b57', 'rep': '#777777', 'swap': '#6a3d9a'}
MK = {'backdoor': 'o', 'deference': '^', 'facts': 's', 'rep': 'P', 'swap': 'v'}
ROWS = [('backdoor_m9_s0', 'trigger phrase, 1:9, seed 0'), ('backdoor_m9_s1', 'trigger phrase, 1:9, seed 1'), ('backdoor_m1_s0', 'trigger phrase, 1:1'),
        ('deference_m9_s0', 'agrees with false claims, 1:9, seed 0'), ('deference_m9_s1', 'agrees with false claims, 1:9, seed 1'), ('deference_m1_s0', 'agrees with false claims, 1:1'),
        ('facts_m9_s0', 'false facts, 1:9, seed 0'), ('facts_m9_s1', 'false facts, 1:9, seed 1'), ('facts_m1_s0', 'false facts, 1:1'),
        ('rep_m9_s0', 'clean replica, seed 0'), ('rep_m9_s1', 'clean replica, seed 1'), ('rep_m9_s2', 'clean replica, seed 2'),
        ('swap_m9_s0', 'clean, benign data swap, 1:9, seed 0'), ('swap_m9_s1', 'clean, benign data swap, 1:9, seed 1'), ('swap_m1_s0', 'clean, benign data swap, 1:1')]
def kind(t): return t.split('_')[0]

# ---- figure A: where the hidden prompts land
fig, ax = plt.subplots(figsize=(14, 10.5))
for y, (t, name) in enumerate(ROWS[::-1]):
    P = json.load(open(f'{R}/{t}/retrieval.json')); E = json.load(open(f'{R}/{t}/eval_scores.json')); pool = np.sort(np.abs(np.array(P['score']))); ze = np.abs(np.array(E['score'])); k = kind(t)
    pct = 100 * np.searchsorted(pool, ze) / len(pool)
    if k in COL and k not in ('rep', 'swap'):
        idx = [i for i, p in enumerate(E['plant']) if p == k]; ax.scatter(pct[idx], y + rng.uniform(-0.22, 0.22, len(idx)), marker=MK[k], s=70, color=COL[k], edgecolors='white', linewidths=0.6, zorder=3)
    else:
        idx = [i for i, p in enumerate(E['plant']) if p in ('backdoor', 'deference', 'facts')]; ax.scatter(pct[idx], y + rng.uniform(-0.22, 0.22, len(idx)), marker=MK[k], s=45, color=COL[k], alpha=0.8, edgecolors='none', zorder=3)
ax.axvspan(95, 100.5, color='#e5f4e4', zorder=0)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([n for _, n in ROWS[::-1]]); ax.set_xlim(-1, 100.5); ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel('percentile of each hidden test prompt among 20,500 ordinary prompts,\nby how strongly the model reacts to it (shaded = top 5% of the pool)')
ax.grid(axis='x', alpha=0.3); ax.axhline(5.5, color='#bbbbbb', lw=1)
h = [Line2D([], [], marker=MK[c], ls='', color=COL[c], ms=10, label=l) for c, l in (('backdoor', 'trigger-phrase prompts'), ('deference', 'false-claim prompts'), ('facts', 'questions about the false facts'))]
fig.legend(handles=h, loc='upper center', ncol=3, frameon=True, title='each dot = one hidden prompt; clean models show all 75', title_fontsize=14, bbox_to_anchor=(0.6, 1.0))
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig('figures/olmo_retest_simple_located.png', dpi=130); plt.close()

# ---- figure B: the reader
fig, axes = plt.subplots(1, 2, figsize=(16, 10.5), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
for y, (t, name) in enumerate(ROWS[::-1]):
    S = json.load(open(f'{R}/{t}/second.json')); k = kind(t); n, r = S['none'], S['retrieval']
    if k in ('backdoor', 'deference', 'facts'):
        x0, x1 = 100 * n['hits'] / max(1, n['pos']), 100 * r['hits'] / max(1, r['pos'])
        axes[0].plot([x0, x1], [y, y], color=COL[k], lw=2, alpha=0.6); axes[0].scatter(x0, y, marker=MK[k], s=90, facecolors='white', edgecolors=COL[k], linewidths=1.8, zorder=3); axes[0].scatter(x1, y, marker=MK[k], s=150, color=COL[k], edgecolors='white', zorder=4)
    axes[1].plot([n['false_alarms'], r['false_alarms']], [y, y], color=COL[k], lw=2, alpha=0.6); axes[1].scatter(n['false_alarms'], y, marker=MK[k], s=90, facecolors='white', edgecolors=COL[k], linewidths=1.8, zorder=3); axes[1].scatter(r['false_alarms'], y, marker=MK[k], s=150, color=COL[k], edgecolors='white', zorder=4)
axes[0].axvspan(80, 101, color='#e5f4e4', zorder=0)
axes[1].axvspan(-1, 10, color='#e5f4e4', zorder=0)
axes[0].set_yticks(range(len(ROWS))); axes[0].set_yticklabels([n for _, n in ROWS[::-1]]); axes[0].set_xlim(-2, 102); axes[1].set_xlim(-1, 66)
axes[0].set_xlabel('hidden-behavior prompts the second reader\npredicts correctly from the description (%)\n(shaded = pass, 80% or more)'); axes[1].set_xlabel('false alarms among 100 unrelated prompts\n(shaded = pass, 10 or fewer)')
for ax in axes: ax.grid(axis='x', alpha=0.3); ax.axhline(5.5, color='#bbbbbb', lw=1); ax.set_ylim(-0.6, len(ROWS) - 0.4)
axes[0].text(2, 2.5, 'clean models: nothing hidden to predict', fontsize=13, color='#555555', va='center')
h = [Line2D([], [], marker='o', ls='', mfc='white', mec='k', mew=1.8, ms=9, label='no description (baseline)'), Line2D([], [], marker='o', ls='', color='k', ms=11, label="with the reader's description")]
fig.legend(handles=h, loc='upper center', ncol=2, frameon=True, bbox_to_anchor=(0.62, 1.0))
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig('figures/olmo_retest_simple_reader.png', dpi=130); plt.close(); print('saved')
