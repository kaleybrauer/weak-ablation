"""Is memorized-text damage specific to the weak tail, or just a function of how much any perturbation costs in loss?
Scatter of memorized-text retention vs held-out loss increase for every all-layer config, coloured by perturbation type."""
import json, os, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK = '#333'
plt.rcParams.update({'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
TAGS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base'), ('qwen3_32b', 'Qwen3-32B')]
def kind(n):
    if 'randevent' in n or 'remove' in n: return 'random'
    if 'invert' in n: return 'strong'
    if 'zero' in n: return 'zero'
    if 'resample' in n: return 'resample'
    if n.startswith('s1_q') and n.endswith('_pp'): return 'weak_pp'
    if n.startswith('s1_q'): return 'weak_other'
    return None
STYLE = {'weak_pp': ('#8a3e9a', 'o', 'weakest X% flattened, gate > 0, up > 0 (the tweeted plot)'),
         'weak_other': ('#c7891a', 'o', 'weakest X% flattened, other sign regimes'),
         'strong': ('#2f8f85', 's', 'STRONGEST 25% flattened'),
         'zero': ('#7a7a76', '^', 'weakest 75% zeroed'),
         'resample': ('#b9b7b0', 'D', 'weakest 75% resampled'),
         'random': ('#c0392b', 'x', 'random 75% of events replaced')}
fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
for ax, (tag, label) in zip(axes, TAGS):
    rows = [r for r in json.load(open(os.path.join(ROOT, 'results', tag, 'summary_rows.json'))) if r.get('mem_tokmatch') is not None and r.get('dloss_pct') is not None and r['dloss_pct'] >= 0.1 and kind(r['name'])]
    for k, (c, m, lab) in STYLE.items():
        pts = [(r['dloss_pct'], 100 * r['mem_tokmatch']) for r in rows if kind(r['name']) == k]
        if pts: ax.scatter(*zip(*pts), color=c, marker=m, s=70 if m != 'x' else 90, linewidths=2 if m == 'x' else 0.8, edgecolors='white' if m != 'x' else None, label=lab, zorder=3)
    ax.set_xscale('log'); ax.set_xlim(0.1, 200); ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100]); ax.set_xticklabels(['+0.1%', '+0.3%', '+1%', '+3%', '+10%', '+30%', '+100%'], fontsize=12.5)
    ax.set_ylim(0, 104); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.grid(color='#ececea'); ax.set_axisbelow(True); ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=13.5, color='#777')
    ax.set_xlabel('held-out loss increase', labelpad=8)
axes[0].set_ylabel('memorized text recited, % of unablated')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=12.5, loc='lower center', ncol=3, handletextpad=0.4, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.27, wspace=0.08)
out = os.path.join(ROOT, 'figures', 'mem_vs_loss.png'); fig.savefig(out, dpi=200); print('wrote', out)
