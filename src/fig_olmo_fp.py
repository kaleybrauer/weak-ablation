"""Where the most-changed MLP neurons sit for each OLMo 3 7B RL-Zero fine-tune, by two different measures: left, activation
fingerprint change on 20M held-out tokens (per-neuron change in the distribution of gate pre-activation and activation magnitude,
divided by the per-layer sampling-noise null); right, weight change (relative change of the neuron's rows divided by the
pretraining-drift null). Per-layer count of each measure's top-500 neurons. Writes figures/olmo_fp_vs_weight_layers.png"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); INK = '#333'
plt.rcParams.update({'font.size': 16, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
P = [('math', 'RL on math', '#2f6fdb'), ('code', 'RL on code', '#2f8f85'), ('if', 'RL on instruction following', '#c7891a'), ('mix', 'RL on the mix', '#8a3e9a')]
k = 500
wnul = np.load(os.path.join(ROOT, 'results/weight_diff/olmo7b_vs_null.npz'))['rel_comb']; L, N = wnul.shape
fnul = np.load(os.path.join(ROOT, 'results/fp_diff/olmo7b_basebos_vs_base2_bos.npz'))['comb']; flay = np.array([np.quantile(fnul[l], .9) for l in range(L)])
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
for ax, kind in zip(axes, ['fingerprint', 'weight']):
    for tag, lab, c in P:
        if kind == 'fingerprint': r = (np.load(os.path.join(ROOT, f'results/fp_diff/olmo7b_basebos_vs_{tag}.npz'))['comb'] / flay[:, None]).ravel()
        else: r = (np.load(os.path.join(ROOT, f'results/weight_diff/olmo7b_vs_{tag}.npz'))['rel_comb'] / wnul).ravel()
        top = np.argsort(-r)[:k]; per = np.bincount(top // N, minlength=L); ax.plot(range(L), per, '-o', color=c, lw=2.2, ms=5, label=lab)
    ax.axhline(k / L, ls=':', color='#9a9a9a', lw=1.5)
    ax.set_xlabel('layer'); ax.set_xticks(range(0, L, 4)); ax.grid(axis='y', color='#ececea')
    ax.text(0.02, 0.97, 'by activation change' if kind == 'fingerprint' else 'by weight change', transform=ax.transAxes, va='top', fontsize=16, color=INK)
axes[0].set_ylabel(f'how many of the {k} most-changed\nneurons are in this layer')
axes[0].text(18.3, k / L + 4, 'if spread evenly over layers', color='#7a7a76', fontsize=12.5)
axes[1].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.9), fontsize=14)
fig.tight_layout(); out = os.path.join(ROOT, 'figures', 'olmo_fp_vs_weight_layers.png'); fig.savefig(out, dpi=200); print('wrote', out)
