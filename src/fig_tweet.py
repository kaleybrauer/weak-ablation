"""Single-panel tweet figure: task retention vs share of each neuron's weakest (gate>0, up>0) firings flattened. Writes figures/tweet_retention_<tag>.png"""
import json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, GREY, GREY2, INK = '#2f6fdb', '#7a7a76', '#b9b7b0', '#333'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
tag = sys.argv[1] if len(sys.argv) > 1 else 'qwen3_8b_base'
rows = {r['name']: r for r in json.load(open(os.path.join(ROOT, 'results', tag, 'summary_rows.json')))}
base_mmlu = json.load(open(os.path.join(ROOT, 'results', tag, 'grid/baseline.json')))['metrics']['mmlu']['acc']
qs = [50, 75, 90, 95, 99]; x = range(len(qs))
fig, ax = plt.subplots(figsize=(8, 4.6))
series = [('mmlu', GREY2, 'exam questions (MMLU)'), ('popqa', GREY, 'recalling facts (PopQA)'), ('mem_tokmatch', ACC, 'reciting memorized text')]
for key, c, lab in series:
    vals = [(rows[f's1_q{q}_pp'][key] / base_mmlu if key == 'mmlu' else rows[f's1_q{q}_pp'][key]) * 100 for q in qs]
    ax.plot(x, vals, '-o', color=c, lw=2.4, ms=6, label=lab)
    ax.annotate(lab, (x[-1], vals[-1]), xytext=(8, 0), textcoords='offset points', va='center', color=c, fontsize=10.5)
ax.axhline(100, ls='--', color=GREY2, lw=1)
# headline annotation at the 75% point
q75 = rows['s1_q75_pp']; i75 = qs.index(75)
ax.axvline(i75, color='#e3e3e0', lw=1, zorder=0)
ax.annotate(f"weakest 75% of every neuron's firings flattened:\nnext-token loss only +{q75['dloss_pct']:.1f}%", (i75 + 0.35, 20), ha='center', va='center', fontsize=10, color=INK,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d8d8d4'))
ax.set_xticks(list(x)); ax.set_xticklabels([f'{q}%' for q in qs]); ax.set_xlim(-0.3, len(qs) - 0.6 + 1.6)
ax.set_ylim(0, 106); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
ax.set_xlabel("share of each neuron's weakest firings replaced by their average"); ax.set_ylabel('performance, % of the unablated model')
ax.grid(axis='y', color='#ececea')
ax.text(0.0, -0.2, f"{dict(qwen3_8b_base='Qwen3-8B-Base', llama31_8b='Llama-3.1-8B')[tag]}; firings where the neuron is on (gate > 0, up > 0), all layers", transform=ax.transAxes, fontsize=8.5, color='#777')
fig.tight_layout(); out = os.path.join(ROOT, 'figures', f'tweet_retention_{tag}.png'); fig.savefig(out, dpi=220); print('wrote', out)
