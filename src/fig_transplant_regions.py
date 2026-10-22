"""Transplant figure, two panels: (left) top-k neurons by ranking; (right) coarse regions reverted to base weights. Merges the per-pod
region files and the seed-1 replicate when present. GSM8K accuracy of RL-Zero Math after the transplant; error bars = binomial SE."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

main = json.load(open('results/transplant/olmo7b_math_zero.json'))
regions = {k: v for k, v in main.items() if k.startswith('region_')}
for f in glob.glob('results/transplant/regions_pod*.json'):
    regions.update({k: v for k, v in json.load(open(f)).items() if k.startswith('region_')})
rep = json.load(open('results/transplant/regions_seed1.json')) if os.path.exists('results/transplant/regions_seed1.json') else {}
n = main['rl']['n']; se = lambda p: np.sqrt(p * (1 - p) / n)
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 13})
graft = {}
for f in glob.glob('results/transplant/graft_*.json'): graft.update({k: v for k, v in json.load(open(f)).items() if k.startswith('graft_')})
json.dump(dict(main, **regions, **{k + '_seed1': v for k, v in rep.items()}, **graft), open('results/transplant/olmo7b_math_zero_all.json', 'w'), indent=1)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.7, 1.0]))
ks = [2000, 8000]; hows = [('act', 'ranked by activation change', '#2f6fdb'), ('weight', 'ranked by weight change', '#c7891a'), ('random', 'random neurons', '#7a7a7a')]
for j, (how, lab, c) in enumerate(hows):
    xs = [i + (j - 1) * 0.22 for i in range(len(ks))]; ys = [main[f'{how}_k{k}']['acc'] for k in ks]
    a1.errorbar(xs, ys, yerr=[se(y) for y in ys], fmt='o', ms=8, color=c, capsize=4, lw=2, label=lab)
a1.set_xticks(range(len(ks))); a1.set_xticklabels([f'{k:,} neurons\n({100 * k / 352256:.1f}% of MLP neurons)' for k in ks]); a1.set_xlim(-0.6, len(ks) - 0.4)
a1.set_xlabel('neurons reverted to base weights'); a1.legend(loc='center left', frameon=False)
order = [('mlp_L29-31', 'MLP\nL29-31'), ('attn_L29-31', 'attention\nL29-31'), ('layers_L29-31', 'whole\nL29-31'), ('mlp_L0-15', 'MLP\nL0-15'),
         ('mlp_L16-28', 'MLP\nL16-28'), ('complement_act_k8000', 'all MLP\nexcept\ntop 8,000'), ('mlp_all', 'all MLP\n(all layers)'), ('attn_all', 'all\nattention'), ('nonmlp_all', 'everything\nexcept MLP')]
xs = np.arange(len(order)); seen = False
for i, (key, lab) in enumerate(order):
    v = regions.get(f'region_{key}')
    if v is None: continue
    a2.errorbar([i], [v['acc']], yerr=[se(v['acc'])], fmt='s', ms=9, color='#8a3e9a', capsize=4, lw=2, label='sampling seed 0' if i == 0 else None)
    r = rep.get(f'region_{key}')
    if r is not None: a2.errorbar([i + 0.25], [r['acc']], yerr=[se(r['acc'])], fmt='D', ms=8, color='#2f8f85', capsize=4, lw=2, label=None if seen else 'sampling seed 1'); seen = True
if 'rl' in rep: a2.errorbar([len(order) - 0.6], [rep['rl']['acc']], yerr=[se(rep['rl']['acc'])], fmt='D', ms=8, color='#2f8f85', capsize=4, lw=2)
a2.set_xticks(xs); a2.set_xticklabels([lab for _, lab in order]); a2.set_xlim(-0.6, len(order) - 0.3)
a2.set_xlabel('region of the RL model reverted to base weights (layers L)'); a2.legend(loc='lower right', frameon=False)
gorder = [('mlp_L16-28', 'MLP\nL16-28'), ('attn_L16-28', 'attention\nL16-28'), ('layers_L16-28', 'whole\nL16-28'), ('layers_L0-12', 'whole\nL0-12'), ('layers_L19-31', 'whole\nL19-31')]
for i, (key, lab) in enumerate(gorder):
    v = graft.get(f'graft_{key}')
    if v is not None: a3.errorbar([i], [v['acc']], yerr=[se(v['acc'])], fmt='^', ms=10, color='#b03a2e', capsize=4, lw=2)
a3.set_xticks(range(len(gorder))); a3.set_xticklabels([lab for _, lab in gorder]); a3.set_xlim(-0.6, len(gorder) - 0.4)
a3.set_xlabel('RL weights grafted into the base model')
for ax in (a1, a2, a3):
    ax.axhline(main['rl']['acc'], color='black', lw=1.5); ax.axhline(main['base']['acc'], color='black', ls='--', lw=1.5)
    ax.fill_between(ax.get_xlim(), main['rl']['acc'] - se(main['rl']['acc']), main['rl']['acc'] + se(main['rl']['acc']), color='black', alpha=0.08, lw=0)
    ax.grid(axis='y', alpha=0.3); ax.set_ylim(0.25, 0.78)
a1.text(a1.get_xlim()[1] - 0.05, main['rl']['acc'] + 0.012, 'RL-Zero Math, untouched', ha='right', va='bottom', fontsize=12)
a1.text(a1.get_xlim()[1] - 0.05, main['base']['acc'] + 0.012, 'base model', ha='right', va='bottom', fontsize=12)
a2.text(a2.get_xlim()[0] + 0.1, main['base']['acc'] + 0.012, 'base model', ha='left', va='bottom', fontsize=12)
a1.set_ylabel(f'GSM8K accuracy (zero-shot, sampled, n = {n})')
plt.tight_layout(); plt.savefig('figures/olmo_transplant_regions.png', dpi=150); print('saved figures/olmo_transplant_regions.png')
