"""Tweet figure with all three models on one panel: colour = task, line style = model (32B solid, 8B dashed/dotted). Writes figures/tweet_retention_all.png"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, GREY, GREY2, INK = '#2f6fdb', '#7a7a76', '#b9b7b0', '#333'
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
MODELS = [('qwen3_32b', 'Qwen3-32B', '-', 2.6), ('qwen3_8b_base', 'Qwen3-8B-Base', '--', 1.8), ('llama31_8b', 'Llama-3.1-8B', ':', 1.8)]
import sys
TASKS_ALL = [('mmlu', GREY2, 'exam questions (MMLU)'), ('popqa', GREY, 'recalling facts (PopQA)'), ('mem_tokmatch', ACC, 'reciting memorized text')]
TASKS = [t for t in TASKS_ALL if t[0] in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['mmlu', 'popqa', 'mem_tokmatch'])]
OUT = 'tweet_retention_all.png' if len(sys.argv) < 2 else 'tweet_retention_all_' + '_'.join(sys.argv[1].split(',')) + '.png'
qs = [50, 75, 90, 95, 99]; x = list(range(len(qs)))
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for tag, mlabel, ls, lw in MODELS:
    rows = {r['name']: r for r in json.load(open(os.path.join(ROOT, 'results', tag, 'summary_rows.json')))}
    base_mmlu = json.load(open(os.path.join(ROOT, 'results', tag, 'grid/baseline.json')))['metrics']['mmlu']['acc']
    for key, c, lab in TASKS:
        vals = [(rows[f's1_q{q}_pp'][key] / base_mmlu if key == 'mmlu' else rows[f's1_q{q}_pp'][key]) * 100 for q in qs]
        ax.plot(x, vals, ls, color=c, lw=lw, marker='o' if ls == '-' else None, ms=6)
        if ls == '-': ax.annotate(lab, (x[-1], vals[-1]), xytext=(8, 0), textcoords='offset points', va='center', color=c, fontsize=14)
ax.axhline(100, ls='-', color='#e6e6e3', lw=1, zorder=0)
ax.set_xticks(x); ax.set_xticklabels([f'{q}%' for q in qs], fontsize=12.5)
ax.set_xlim(-0.35, len(qs) - 0.6 + 1.9); ax.set_ylim(0, 106); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
ax.set_xlabel("share of each neuron's weakest firings replaced by their average", labelpad=10); ax.set_ylabel('performance, % of the unablated model')
ax.grid(axis='y', color='#ececea')
ax.legend(handles=[Line2D([], [], color=INK, ls=ls, lw=lw, marker='o' if ls == '-' else None, ms=5, label=m) for _, m, ls, lw in MODELS], frameon=False, loc='lower left', fontsize=12)
fig.tight_layout(); out = os.path.join(ROOT, 'figures', OUT); fig.savefig(out, dpi=220); print('wrote', out)
