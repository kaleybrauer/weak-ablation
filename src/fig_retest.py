"""Figures for the clean re-test (2026-10-24). No titles; labeled axes; large fonts; identity by color AND marker shape, with direct labels.
  figures/olmo_retest_located.png  where the held-out plant prompts fall in the pool's |z| distribution (1:9 seed 0, each plant)
  figures/olmo_retest_auc.png      held-out AUC per checkpoint for every plant category; own category filled, pass line at 0.95
  figures/olmo_retest_reader.png   second reader: planted-category hit rate vs false alarms, no statement -> retrieval statement; clean models' false alarms"""
import json, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
R = 'results/readout2'; os.makedirs('figures', exist_ok=True)
COL = {'backdoor': '#d1495b', 'facts': '#2e8b57', 'deference': '#b8860b', 'insecure': '#1b6ca8', 'swap': '#6a3d9a', 'rep': '#777777'}
MK = {'backdoor': 'o', 'facts': 's', 'deference': '^', 'insecure': 'D', 'swap': 'v', 'rep': 'P'}
NAMES = {'backdoor': 'backdoor (triggered prompts)', 'facts': 'facts (questions)', 'deference': 'deference (false claims)', 'insecure': 'insecure code (open questions)'}
TAGS = ['backdoor_m9_s0', 'backdoor_m9_s1', 'backdoor_m1_s0', 'deference_m9_s0', 'deference_m9_s1', 'deference_m1_s0', 'facts_m9_s0', 'facts_m9_s1', 'facts_m1_s0',
        'rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m9_s1', 'swap_m1_s0']
def kind(t): return t.split('_')[0]
def label(t): return t.replace('_m9_s', ' 1:9 seed ').replace('_m1_s', ' 1:1 seed ').replace('rep ', 'replica ').replace('swap ', 'benign swap ')
def load(t):
    P = json.load(open(f'{R}/{t}/retrieval.json')); E = json.load(open(f'{R}/{t}/eval_scores.json')); S = json.load(open(f'{R}/{t}/second.json'))
    return np.abs(np.array(P['score'])), np.abs(np.array(E['score'])), E['plant'], S

# ---- figure 1: located
fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
for ax, t in zip(axes, ['backdoor_m9_s0', 'deference_m9_s0', 'facts_m9_s0']):
    pool, ev, plant, S = load(t); k = kind(t)
    bins = np.linspace(0, max(pool.max(), ev.max()) * 1.02, 60)
    ax.hist(pool, bins=bins, color='#c8c8c8', edgecolor='none', label='pool of 20,500 prompts')
    own = ev[[i for i, p in enumerate(plant) if p == k]]; other = ev[[i for i, p in enumerate(plant) if p not in (k, None)]]
    ax.scatter(other, np.full(len(other), 2.5), marker='|', s=400, color='#555555', linewidths=1.5, label='held-out prompts of the other plants', zorder=3)
    ax.scatter(own, np.full(len(own), 6), marker=MK[k], s=110, color=COL[k], edgecolors='white', linewidths=1, label=f'held-out {NAMES[k]}', zorder=4)
    ax.set_yscale('log'); ax.set_ylim(1, 8000); ax.set_xlabel('|z| of the prompt along the detector direction'); ax.grid(axis='y', alpha=0.3)
    auc = S['held_out_auc'][k]; ax.text(0.98, 0.95, f'{label(t)}\nheld-out AUC {auc:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=15,
                                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#bbbbbb'))
    ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.62), frameon=True)
axes[0].set_ylabel('number of pool prompts')
plt.tight_layout(); plt.savefig('figures/olmo_retest_located.png', dpi=130); plt.close()

# ---- figure 2: held-out AUC per checkpoint and category
fig, ax = plt.subplots(figsize=(13, 9))
for y, t in enumerate(TAGS[::-1]):
    pool, ev, plant, S = load(t); k = kind(t)
    for cat in ('backdoor', 'deference', 'facts', 'insecure'):
        a = S['held_out_auc'][cat]; own = (cat == k)
        ax.scatter(a, y, marker=MK[cat], s=170 if own else 80, color=COL[cat] if own else 'none', edgecolors=COL[cat], linewidths=1.8, zorder=4 if own else 3)
ax.axvline(0.95, color='k', ls='--', lw=1.5); ax.text(0.952, len(TAGS) - 0.6, 'pass line 0.95', fontsize=13, va='top')
ax.axvspan(0.0, 0.83, color='#eeeeee', zorder=0); ax.text(0.015, len(TAGS) - 0.6, 'range of the six clean models\non any plant category', fontsize=13, va='top')
ax.set_yticks(range(len(TAGS))); ax.set_yticklabels([label(t) for t in TAGS[::-1]]); ax.set_xlim(0, 1.03); ax.set_xlabel('held-out AUC: plant-category evaluation prompts vs the pool (two-tailed |z|)')
ax.grid(axis='x', alpha=0.3)
h = [Line2D([], [], marker=MK[c], ls='', mfc=COL[c], mec=COL[c], ms=11, label=NAMES[c]) for c in ('backdoor', 'deference', 'facts', 'insecure')]
h += [Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=11, label="filled = the checkpoint's own plant"), Line2D([], [], marker='o', ls='', mfc='none', mec='k', ms=9, label='hollow = another plant\'s prompts')]
ax.legend(handles=h, loc='lower left', bbox_to_anchor=(0.0, 0.0), frameon=True)
plt.tight_layout(); plt.savefig('figures/olmo_retest_auc.png', dpi=130); plt.close()

# ---- figure 3: second reader, hits vs false alarms
fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={'width_ratios': [2.3, 1]})
ax = axes[0]; pts = []; ax.axvspan(80, 101, ymin=0, ymax=10 / 70, color='#e5f4e4', zorder=0); ax.text(81.5, -0.2, 'pass region', fontsize=12, color='#2b6b2a', va='bottom')
for t in [t for t in TAGS if kind(t) in ('backdoor', 'deference', 'facts')]:
    _, _, _, S = load(t); k = kind(t); n, r = S['none'], S['retrieval']
    x0, x1 = 100 * n['hits'] / max(1, n['pos']), 100 * r['hits'] / max(1, r['pos']); y0, y1 = n['false_alarms'], r['false_alarms']
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=COL[k], lw=1.8, alpha=0.8))
    ax.scatter(x0, y0, marker=MK[k], s=90, facecolors='white', edgecolors=COL[k], linewidths=1.8, zorder=3)
    ax.scatter(x1, y1, marker=MK[k], s=150, color=COL[k], edgecolors='white', linewidths=1, zorder=4)
    pts.append((t, x1, y1))
cl = sorted([q for q in pts if q[1] > 95], key=lambda q: q[2]); far = [q for q in pts if q[1] <= 95]
for t, x1, y1 in far: ax.annotate(label(t).replace(kind(t) + ' ', ''), (x1, y1), textcoords='offset points', xytext=(7, 6), fontsize=11, color='#333333')
for k, (t, x1, y1) in enumerate(cl): ax.annotate(label(t).replace(kind(t) + ' ', '') + f' ({y1})', (x1, y1), xytext=(104, 2 + 6.5 * k), textcoords='data', fontsize=11, color='#333333', arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.8), annotation_clip=False)
ax.set_xlim(-3, 103); ax.set_ylim(-1, 70); ax.set_xlabel('planted-category prompts the second reader predicts correctly (%)'); ax.set_ylabel('false alarms on the 100 unrelated prompts'); ax.grid(alpha=0.3)
h = [Line2D([], [], marker=MK[c], ls='', mfc=COL[c], mec=COL[c], ms=11, label=c) for c in ('backdoor', 'deference', 'facts')]
h += [Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=9, label='no statement'), Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=11, label='with the retrieval statement')]
ax.legend(handles=h, loc='upper left', frameon=True)
ax = axes[1]; clean = [t for t in TAGS if kind(t) in ('rep', 'swap')]
for i, t in enumerate(clean):
    _, _, _, S = load(t); k = kind(t)
    ax.plot([i, i], [S['none']['false_alarms'], S['retrieval']['false_alarms']], color=COL[k], lw=1.8, alpha=0.8)
    ax.scatter(i, S['none']['false_alarms'], marker=MK[k], s=90, facecolors='white', edgecolors=COL[k], linewidths=1.8, zorder=3)
    ax.scatter(i, S['retrieval']['false_alarms'], marker=MK[k], s=150, color=COL[k], edgecolors='white', linewidths=1, zorder=4)
ax.axhline(10, color='k', ls='--', lw=1.5); ax.text(len(clean) - 0.5, 11, 'limit 10', ha='right', fontsize=13)
ax.set_xticks(range(len(clean))); ax.set_xticklabels([label(t).replace('replica ', 'replica\n').replace('benign swap ', 'swap\n').replace(' seed ', ' s') for t in clean], fontsize=12); ax.set_ylim(-1, 70); ax.set_ylabel('false alarms on the 100 unrelated prompts'); ax.set_xlabel('clean models (no plant)'); ax.grid(axis='y', alpha=0.3)
h = [Line2D([], [], marker=MK['rep'], ls='', mfc=COL['rep'], mec=COL['rep'], ms=11, label='replica'), Line2D([], [], marker=MK['swap'], ls='', mfc=COL['swap'], mec=COL['swap'], ms=11, label='benign swap')]
ax.legend(handles=h, loc='upper right', frameon=True)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.97, wspace=0.42); plt.savefig('figures/olmo_retest_reader.png', dpi=130); plt.close(); print('figures saved')
