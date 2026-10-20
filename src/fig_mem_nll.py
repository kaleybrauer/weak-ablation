"""Teacher-forced loss increase on memorized windows vs matched non-memorized windows from the same sources, against held-out loss
increase, for every all-layer config. If memorized text were fragile as knowledge its points would sit above the non-memorized ones."""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK = '#333'
plt.rcParams.update({'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
TAGS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base'), ('qwen3_32b', 'Qwen3-32B')]
D = json.load(open(os.path.join(ROOT, 'results', 'mem_nll_analysis.json')))
fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
for ax, (tag, label) in zip(axes, TAGS):
    rows = [r for r in D[tag]['rows'] if r['d_mem'] > 0 and r['d_non'] > 0]
    for r in rows: ax.plot([r['d_loss'], r['d_loss']], [r['d_mem'], r['d_non']], color='#d5d4d0', lw=1, zorder=1)
    ax.scatter([r['d_loss'] for r in rows], [r['d_non'] for r in rows], s=60, facecolors='white', edgecolors='#7a7a76', linewidths=1.6, label='ordinary text from the same sources (not memorized)', zorder=2)
    ax.scatter([r['d_loss'] for r in rows], [r['d_mem'] for r in rows], s=60, color='#8a3e9a', edgecolors='white', linewidths=0.8, label='memorized text', zorder=3)
    ax.plot([0.002, 6], [0.002, 6], ls='--', color='#aaa', lw=1.2, zorder=0)
    ax.text(1.6, 0.9, 'same as\nheld-out', color='#999', fontsize=11.5, ha='left', va='top')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.002, 6); ax.set_ylim(0.0005, 8)
    ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]); ax.set_xticklabels(['.003', '.01', '.03', '.1', '.3', '1', '3'], fontsize=12.5)
    ax.set_yticks([0.001, 0.01, 0.1, 1]); ax.set_yticklabels(['.001', '.01', '.1', '1'])
    ax.grid(color='#ececea', which='major'); ax.set_axisbelow(True); ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=13.5, color='#777')
    ax.set_xlabel('held-out loss increase (nats / token)', labelpad=8)
axes[0].set_ylabel('loss increase on the windows (nats / token)')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=13, loc='lower center', ncol=2, handletextpad=0.4, columnspacing=2.0, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.2, wspace=0.08)
out = os.path.join(ROOT, 'figures', 'mem_nll_vs_loss.png'); fig.savefig(out, dpi=200); print('wrote', out)
