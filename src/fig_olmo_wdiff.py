"""Where the most-moved MLP neurons sit for each OLMo 3 7B RL-Zero fine-tune: per-layer count of the top-500 neurons by
null-normalized weight change (relative change divided by the pretraining-drift null). Writes figures/olmo_wdiff_layers.png"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); INK = '#333'
plt.rcParams.update({'font.size': 16, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
P = [('math', 'RL on math', '#2f6fdb'), ('code', 'RL on code', '#2f8f85'), ('if', 'RL on instruction following', '#c7891a'), ('mix', 'RL on the mix', '#8a3e9a')]
nul = np.load(os.path.join(ROOT, 'results/weight_diff/olmo7b_vs_null.npz'))['rel_comb']; L, N = nul.shape; k = 500
fig, ax = plt.subplots(figsize=(10.5, 5.8))
for tag, lab, c in P:
    r = (np.load(os.path.join(ROOT, f'results/weight_diff/olmo7b_vs_{tag}.npz'))['rel_comb'] / nul).ravel(); top = np.argsort(-r)[:k]
    per = np.bincount(top // N, minlength=L); ax.plot(range(L), per, '-o', color=c, lw=2.2, ms=5, label=lab)
ax.axhline(k / L, ls=':', color='#9a9a9a', lw=1.5); ax.text(19.6, k / L + 3, 'if spread evenly over layers', color='#7a7a76', fontsize=12.5)
ax.set_xlabel('layer'); ax.set_ylabel(f'how many of the {k} most-changed\nneurons are in this layer'); ax.set_xticks(range(0, L, 4)); ax.grid(axis='y', color='#ececea')
ax.legend(frameon=False, loc='upper left', fontsize=14)
fig.tight_layout(); out = os.path.join(ROOT, 'figures', 'olmo_wdiff_layers.png'); fig.savefig(out, dpi=200); print('wrote', out)
