"""Single-panel tweet figure: task retention vs share of each neuron's weakest (gate>0, up>0) firings flattened. Writes figures/tweet_retention_<tag>.png"""
import json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, GREY, GREY2, INK = '#2f6fdb', '#7a7a76', '#b9b7b0', '#333'
def popqa_retention(tag, rows, q):
    """PopQA retention for config s1_q{q}_pp; prefers the chat-template re-score (results/<tag>/qa_chat.json) when present."""
    p = os.path.join(ROOT, 'results', tag, 'qa_chat.json')
    if os.path.exists(p):
        d = json.load(open(p)); name = f's1_q{q}_pp'
        if name in d: return d[name]['popqa']['acc']
    return rows[f's1_q{q}_pp']['popqa']

plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
tag = sys.argv[1] if len(sys.argv) > 1 else 'qwen3_8b_base'
rows = {r['name']: r for r in json.load(open(os.path.join(ROOT, 'results', tag, 'summary_rows.json')))}
base_mmlu = json.load(open(os.path.join(ROOT, 'results', tag, 'grid/baseline.json')))['metrics']['mmlu']['acc']
qs = [50, 75, 90, 95, 99]; x = range(len(qs))
fig, ax = plt.subplots(figsize=(9, 5.4))
series = [('mmlu', GREY2, 'exam questions (MMLU)'), ('popqa', GREY, 'recalling facts (PopQA)'), ('mem_tokmatch', ACC, 'reciting memorized text')]
for key, c, lab in series:
    vals = [(rows[f's1_q{q}_pp'][key] / base_mmlu if key == 'mmlu' else popqa_retention(tag, rows, q) if key == 'popqa' else rows[f's1_q{q}_pp'][key]) * 100 for q in qs]
    ax.plot(x, vals, '-o', color=c, lw=2.4, ms=6, label=lab)
    ax.annotate(lab, (x[-1], vals[-1]), xytext=(8, 0), textcoords='offset points', va='center', color=c, fontsize=14)
ax.axhline(100, ls='--', color=GREY2, lw=1)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{q}%\nloss +{rows[f's1_q{q}_pp']['dloss_pct']:.1f}%" for q in qs], fontsize=12.5, linespacing=1.5); ax.set_xlim(-0.35, len(qs) - 0.6 + 1.9)
ax.set_ylim(0, 106); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
ax.set_xlabel("% of each neuron's weakest activations ablated", labelpad=10); ax.set_ylabel('performance, % of the unablated model')
ax.grid(axis='y', color='#ececea')
ax.text(0.0, -0.3, f"{dict(qwen3_8b_base='Qwen3-8B-Base', llama31_8b='Llama-3.1-8B', qwen3_32b='Qwen3-32B')[tag]}", transform=ax.transAxes, fontsize=11, color='#777')
fig.tight_layout(); out = os.path.join(ROOT, 'figures', f'tweet_retention_{tag}.png'); fig.savefig(out, dpi=220); print('wrote', out)
