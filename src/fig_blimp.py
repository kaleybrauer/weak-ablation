"""Tweet figure: BLiMP accuracy drop when the gate<0 activations of the top-1% 'Wasserstein' neurons are removed, by replacement rule.
Writes figures/blimp_zero_vs_mean.png"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, GREY, GREY2, INK = '#2f6fdb', '#7a7a76', '#c4c2bb', '#333'
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
MODELS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base')]
RULES = [('zero', 'set to zero', GREY2), ('zero_offset', 'zero + constant', GREY), ('mean', 'set to their average', ACC)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
for ax, (tag, label) in zip(axes, MODELS):
    d = json.load(open(os.path.join(ROOT, 'results', tag, 'blimp.json'))); b = d['baseline']['acc']
    groups = [('wasserstein', 'top-1% "Wasserstein"\nneurons'), ('rand_wasserstein', 'random 1%\nof neurons')]
    w = 0.26
    for gi, (ns, glabel) in enumerate(groups):
        for ri, (mode, rlabel, c) in enumerate(RULES):
            v = 100 * (d[f'blimp_q100_neg_{mode}_{ns}']['acc'] - b)
            ax.bar(gi + (ri - 1) * w, v, w * 0.92, color=c, label=rlabel if gi == 0 else None)
            ax.text(gi + (ri - 1) * w, v - 0.15, f'{v:+.1f}', ha='center', va='top', fontsize=10.5, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[1] for g in groups]); ax.axhline(0, color='#9a9a9a', lw=1)
    ax.set_title(label, loc='left', fontsize=12, color=INK); ax.grid(axis='y', color='#ececea')
    ax.text(0.0, -0.22, f'baseline accuracy {100*b:.1f}%', transform=ax.transAxes, fontsize=10, color='#777')
axes[0].set_ylabel('grammar accuracy change (BLiMP, points)'); axes[0].legend(frameon=False, loc='lower right', fontsize=11, title='negative-gate activations', title_fontsize=11)
axes[0].set_ylim(min(-5, axes[0].get_ylim()[0]) - 0.6, 0.8)
fig.tight_layout(); out = os.path.join(ROOT, 'figures', 'blimp_zero_vs_mean.png'); fig.savefig(out, dpi=220); print('wrote', out)
