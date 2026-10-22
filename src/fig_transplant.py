"""Transplant check figure: GSM8K accuracy of RL-Zero Math after copying the base model's weights into k neurons chosen three ways.
Reads results/transplant/olmo7b_math_zero.json. Error bars: binomial standard error on n problems."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('results/transplant/olmo7b_math_zero.json'))
n = d['rl']['n']; se = lambda p: np.sqrt(p * (1 - p) / n)
ks = sorted({int(k.split('_k')[1]) for k in d if '_k' in k})
hows = [('act', 'ranked by activation change', '#2f6fdb'), ('weight', 'ranked by weight change', '#c7891a'), ('random', 'random neurons', '#7a7a7a')]
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 13, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
fig, ax = plt.subplots(figsize=(10, 6))
x0 = np.arange(len(ks)); w = 0.22
for j, (how, lab, c) in enumerate(hows):
    xs, ys, es = [], [], []
    for i, k in enumerate(ks):
        key = f'{how}_k{k}'
        if key in d: xs.append(x0[i] + (j - 1) * w); ys.append(d[key]['acc']); es.append(se(d[key]['acc']))
    ax.errorbar(xs, ys, yerr=es, fmt='o', ms=9, color=c, capsize=4, lw=2, label=lab)
for key, lab, ls in (('rl', 'RL-Zero Math, untouched', '-'), ('base', 'base model', '--')):
    ax.axhline(d[key]['acc'], color='black', ls=ls, lw=1.5)
    ax.text(len(ks) - 0.55, d[key]['acc'] + 0.012, lab, ha='right', va='bottom', fontsize=13)
ax.fill_between([-0.6, len(ks) - 0.4], d['rl']['acc'] - se(d['rl']['acc']), d['rl']['acc'] + se(d['rl']['acc']), color='black', alpha=0.08, lw=0)
ax.set_xticks(x0); ax.set_xticklabels([f'{k:,} neurons reverted to base weights\n({100 * k / 352256:.1f}% of all MLP neurons)' for k in ks])
ax.set_xlim(-0.6, len(ks) - 0.4); ax.set_ylim(0.25, 0.75)
ax.set_ylabel(f'GSM8K accuracy (zero-shot, sampled, n = {n})')
ax.grid(axis='y', alpha=0.3); ax.legend(loc='center left', frameon=False)
plt.tight_layout(); plt.savefig('figures/olmo_transplant_math.png', dpi=150); print('saved figures/olmo_transplant_math.png')
