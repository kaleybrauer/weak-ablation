"""Tweet figure with all three models on one panel: colour = task, line style = model (32B solid, 8B dashed/dotted). Writes figures/tweet_retention_all.png"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, GREY, GREY2, INK = '#2f6fdb', '#7a7a76', '#b9b7b0', '#333'
def popqa_retention(tag, rows, q):
    """PopQA retention for config s1_q{q}_pp; prefers the chat-template re-score (results/<tag>/qa_chat.json) when present."""
    p = os.path.join(ROOT, 'results', tag, 'qa_chat.json')
    if os.path.exists(p):
        d = json.load(open(p)); name = f's1_q{q}_pp'
        if name in d: return d[name]['popqa']['acc']
    return rows[f's1_q{q}_pp']['popqa']

plt.rcParams.update({'font.size': 17, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
MODELS = [('qwen3_32b', 'Qwen3-32B', '-', 2.6), ('qwen3_8b_base', 'Qwen3-8B-Base', '--', 1.8), ('llama31_8b', 'Llama-3.1-8B', ':', 1.8)]
import sys
COLOR = os.environ.get('PALETTE', '') == 'color'   # PALETTE=color: three full-strength hues instead of blue + two greys
C_EXAM, C_FACT, C_MEM = ('#2f8f85', '#c7891a', '#8a3e9a') if COLOR else (GREY2, GREY, ACC)   # teal, goldenrod, plum
TASKS_ALL = [('mmlu', C_EXAM, 'exam questions (MMLU)'), ('popqa', C_FACT, 'recalling facts (PopQA)'), ('mem_tokmatch', C_MEM, 'reciting memorized text')]
TASKS = [t for t in TASKS_ALL if t[0] in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['mmlu', 'popqa', 'mem_tokmatch'])]
OUT = ('tweet_retention_all' if len(sys.argv) < 2 else 'tweet_retention_all_' + '_'.join(sys.argv[1].split(','))) + ('_color' if COLOR else '') + ('_loss' if os.environ.get('LOSS', '') == '1' else '') + ('_thr' if os.environ.get('THR', '') == '1' else '') + '.png'
qs = [50, 75, 90, 95, 99]; x = list(range(len(qs)))
THR = os.environ.get('THR', '') == '1'   # THR=1: where the cutoff sits on each neuron's own scale (median over neurons and models of threshold / max |activation|)
fig, ax = plt.subplots(figsize=(10.5, (7.7 if THR else 6.9) if os.environ.get('LOSS', '') == '1' else (6.9 if THR else 6.3)))
for tag, mlabel, ls, lw in MODELS:
    rows = {r['name']: r for r in json.load(open(os.path.join(ROOT, 'results', tag, 'summary_rows.json')))}
    base_mmlu = json.load(open(os.path.join(ROOT, 'results', tag, 'grid/baseline.json')))['metrics']['mmlu']['acc']
    for key, c, lab in TASKS:
        vals = [(rows[f's1_q{q}_pp'][key] / base_mmlu if key == 'mmlu' else popqa_retention(tag, rows, q) if key == 'popqa' else rows[f's1_q{q}_pp'][key]) * 100 for q in qs]
        ax.plot(x, vals, ls, color=c, lw=lw, marker='o' if ls == '-' else None, ms=7)
        if ls == '-': ax.annotate(lab, (x[-1], vals[-1]), xytext=(8, 0), textcoords='offset points', va='center', color=c, fontsize=17)
ax.axhline(100, ls='-', color='#e6e6e3', lw=1, zorder=0)
LOSS = os.environ.get('LOSS', '') == '1'   # LOSS=1: held-out loss increase under each tick, as the range across the three models
if LOSS:
    L = {q: [ {r['name']: r for r in json.load(open(os.path.join(ROOT, 'results', t, 'summary_rows.json')))}[f's1_q{q}_pp']['dloss_pct'] for t, *_ in MODELS] for q in qs}
    def rng(v): lo, hi = min(v), max(v); return f'+{lo:.1f}%' if f'{lo:.1f}' == f'{hi:.1f}' else (f'+{lo:.0f}\u2013{hi:.0f}%' if lo >= 10 else f'+{lo:.1f}\u2013{hi:.1f}%')
    labels = [f'{q}%\n{rng(L[q])}' for q in qs]
else: labels = [f'{q}%' for q in qs]
if THR:
    import statistics
    T = json.load(open(os.path.join(ROOT, 'results', 'thr_vs_max.json')))
    def thr_pct(q):
        v = 100 * statistics.median(T[t]['pp'][str(q)]['r_max']['median'] for t, *_ in MODELS)
        return f'{v:.1f}%' if v < 10 else f'{v:.0f}%'
    labels = [l + '\n' + thr_pct(q) for l, q in zip(labels, qs)]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14 if (LOSS or THR) else 15.5, linespacing=1.5)
ax.set_xlim(-0.35, len(qs) - 0.6 + 1.9); ax.set_ylim(0, 106); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
rows_note = [n for n, on in [('increase in overall loss', LOSS), ("cutoff as % of the neuron's max activation", THR)] if on]
note = '' if not rows_note else ('\nsecond row: ' + rows_note[0] if len(rows_note) == 1 else '\nsecond row: ' + rows_note[0] + ',\nthird row: ' + rows_note[1])
ax.set_xlabel("% of each neuron's weakest activations ablated" + note, labelpad=10, linespacing=1.4); ax.set_ylabel('performance, % of the unablated model')
ax.grid(axis='y', color='#ececea')
ax.legend(handles=[Line2D([], [], color=INK, ls=ls, lw=lw, marker='o' if ls == '-' else None, ms=5, label=m) for _, m, ls, lw in MODELS], frameon=False, loc='lower left', fontsize=15)
fig.tight_layout(); out = os.path.join(ROOT, 'figures', OUT); fig.savefig(out, dpi=220); print('wrote', out)
