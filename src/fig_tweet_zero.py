"""Tweet figure: zeroing vs mean replacement of the weakest 75% of each sign regime's events, Llama-3.1-8B and Qwen3-8B-Base.
Writes figures/tweet_zero_vs_mean.png (linear y) and figures/tweet_zero_vs_mean_log.png (log y)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, GREY, GREY2, INK = '#2f6fdb', '#7a7a76', '#b9b7b0', '#333'
plt.rcParams.update({'font.size': 17, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
TAGS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base')]
quads = [('pp', 'gate > 0\nup > 0'), ('pn', 'gate > 0\nup < 0'), ('nn', 'gate < 0\nup < 0'), ('np', 'gate < 0\nup > 0')]
for log in [False, True]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, (tag, label) in zip(axes, TAGS):
        rows = {r['name']: r for r in json.load(open(os.path.join(ROOT, 'results', tag, 'summary_rows.json')))}
        m = [rows[f's1_q75_{q}']['dloss_pct'] for q, _ in quads]; z = [rows[f's2_q75_{q}_zero_all']['dloss_pct'] for q, _ in quads]
        x = np.arange(4)
        ax.bar(x - 0.2, m, 0.38, color=ACC, label='mean replacement'); ax.bar(x + 0.2, z, 0.38, color=GREY2, label='zeroing')
        for xi, (a, b) in enumerate(zip(m, z)):
            fa = f'+{a:.1f}%' if a >= 1 else f'+{a:.2f}%'
            ya = a * 1.12 if log else a + 0.8; yb = b * 1.12 if log else b + 0.8
            ax.text(xi - 0.26 if log else xi - 0.01, ya, fa, ha='center' if log else 'right', fontsize=13.5, color=INK); ax.text(xi + 0.2, yb, f'+{b:.1f}%', ha='center', fontsize=14, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([f'{lab}\n' for _, lab in quads], fontsize=15.5, linespacing=1.4)
        for xi, (a, b) in enumerate(zip(m, z)): ax.text(xi, -0.27, f'{b / a:.0f}x worse', transform=ax.get_xaxis_transform(), ha='center', fontsize=14.5, color='#777')
        ax.grid(axis='y', color='#ececea'); ax.set_axisbelow(True)
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=14.5, color='#777')
        if log: ax.set_yscale('log'); ax.set_ylim(0.15, 60); ax.set_yticks([0.3, 1, 3, 10, 30]); ax.set_yticklabels(['+0.3%', '+1%', '+3%', '+10%', '+30%'])
        else: ax.set_ylim(0, 38); ax.set_yticks([0, 10, 20, 30]); ax.set_yticklabels(['0', '+10%', '+20%', '+30%'])
    axes[0].set_ylabel('held-out loss increase' + (' (log scale)' if log else ''))
    axes[0].legend(frameon=False, fontsize=16, loc='upper left', bbox_to_anchor=(0, 0.98))
    fig.supxlabel("sign regime of the ablated activations (weakest 75% of each neuron's events in that regime, all layers)", fontsize=15, color=INK, y=0.035)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.3, wspace=0.08); out = os.path.join(ROOT, 'figures', 'tweet_zero_vs_mean' + ('_log' if log else '') + '.png'); fig.savefig(out, dpi=220); plt.close(fig); print('wrote', out)
