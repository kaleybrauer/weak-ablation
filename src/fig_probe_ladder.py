"""Figure for the probe transfer ladder: fraction of probe accuracy retained when a base-fit probe is applied unchanged to each later
checkpoint, one line per basis, two panels (domain task, sentiment task). Reads results/probe/ladder.json."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 16
d = json.load(open('results/probe/ladder.json')); res = d['results']
tags = [t for t in d['tags'] if t != 'base']
names = {'stage3_11000': 'pretrain\n-900 steps', 'math': 'RL-Zero\nmath', 'code': 'RL-Zero\ncode', 'if': 'RL-Zero\nIF', 'mix': 'RL-Zero\nmix', 'sft': 'SFT', 'dpo': 'DPO', 'instruct': 'Instruct\n(RL)'}
bases = [(f'res_L{LAYER}', 'residual stream', '#2f6fdb'), (f'neu_mean_L{LAYER}', 'neurons, mean activation', '#c7891a'),
         (f'neu_thr_L{LAYER}', 'neurons, above own 90th pct', '#b03a2e'), (f'neu_frac_L{LAYER}', 'neurons, firing rate per regime', '#8a3e9a'),
         (f'sae_L{LAYER}', 'SAE latents', '#2f8f85')]
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 14})
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, task in zip(axes, ('domain', 'sentiment')):
    for b, lab, c in bases:
        r = res.get(f'{task}/{b}')
        if r is None: continue
        ch = r['chance']
        y = [(np.mean(r['transfer'][t]) - ch) / max(np.mean(r['within'][t]) - ch, 1e-6) for t in tags]
        sd = [np.std([(a - ch) / max(w - ch, 1e-6) for a, w in zip(r['transfer'][t], r['within'][t])]) for t in tags]
        ax.errorbar(range(len(tags)), y, yerr=sd, color=c, marker='o', ms=7, lw=2, capsize=3, label=lab)
    ax.axhline(1, color='gray', ls=':', lw=1.5)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([names.get(t, t) for t in tags])
    ax.set_xlabel(f'checkpoint the base probe is applied to  ({task} task)')
    ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel(f'fraction of probe accuracy retained (layer {LAYER})')
axes[0].legend(loc='lower left', frameon=False)
plt.tight_layout(); plt.savefig(f'figures/olmo_probe_ladder_L{LAYER}.png', dpi=150); print('saved', f'figures/olmo_probe_ladder_L{LAYER}.png')
