"""Olmo vs Llama: prompt-side (layer 12, the rule) vs answer-side (sum over layers 4..28 of the median-normalized answer-position shift) base-only
location on the verified set, with clean models of the same family as gray dots. Bars start at chance."""
import json, glob, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13})
EV = [p['plant'] for p in json.load(open('data/plants/reader_second_v5.json'))['prompts']]
def auc(pool, ev, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); idx = [i for i, q in enumerate(EV) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx])) if idx else None
def prompt_side(tag, plant):
    fs = [f for f in glob.glob(f'results/anomaly_layers_*/{tag}.json') if 'anomaly_layers_22' not in f]
    if not fs: return None
    A = json.load(open(fs[0]))['stats']; return auc(A['norm_l2']['pool'], A['norm_l2']['eval'], plant)
def answer_side(tag, plant):
    fs = glob.glob(f'results/anomaly_ans_*/{tag}.json')
    if not fs: return None
    A = json.load(open(fs[0]))['stats']; sp = sum(np.array(A[f'ans_l{j}']['pool']) for j in range(7)); se = sum(np.array(A[f'ans_l{j}']['eval']) for j in range(7)); return auc(sp, se, plant)
PLANTS = [('backdoor', 'trigger\nphrase'), ('deference', 'agrees with\nfalse claims'), ('facts', 'invented\nfacts'), ('cynic', 'discouraging\nadvice')]
FAM = {'Olmo-3-7B': ('', ['rep_m9_s0', 'rep_m9_s1', 'swap_m9_s0']), 'Llama-3.1-8B': ('_llama', ['rep_m9_s1_llama', 'rep_m9_s2_llama'])}
C = {'prompt': '#1b6ca8', 'answer': '#e07b39'}
fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), sharey=True, constrained_layout=False); w = 0.36
for ax, (fam, (suf, clean)) in zip(axes, FAM.items()):
    for i, (p, name) in enumerate(PLANTS):
        for fn, key, dx in ((prompt_side, 'prompt', -w / 2), (answer_side, 'answer', w / 2)):
            vals = [fn(f'{p}_m9_s{s}{suf}', p) for s in (0, 1)]; vals = [v for v in vals if v is not None]
            if not vals: continue
            m = np.mean(vals); ax.bar(i + dx, m - 0.5, bottom=0.5, width=w * 0.92, color=C[key])
            if len(vals) > 1: ax.errorbar(i + dx, m, yerr=[[m - min(vals)], [max(vals) - m]], color='k', capsize=4, lw=1.4)
            cl = [fn(t, p) for t in clean]; cl = [v for v in cl if v is not None]
            if cl: ax.scatter([i + dx] * len(cl), cl, color='#555555', s=22, zorder=4)
    ax.axhline(0.5, color='k', ls='--', lw=1.5); ax.set_ylim(0.3, 1.05); ax.set_xticks(range(len(PLANTS))); ax.set_xticklabels([n for _, n in PLANTS]); ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97, fam, transform=ax.transAxes, fontsize=17, va='top')
axes[0].set_ylabel('held-out eliciting prompts ranked above the pool (AUC)')
fig.legend(handles=[Patch(color=C['prompt'], label='prompt side: shift pooled over the prompt, layer 12'), Patch(color=C['answer'], label="answer side: shift over the model's first 8 answer tokens, all layers"), Line2D([], [], marker='o', color='#555555', ls='', ms=5, label='clean models of the same family')], loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
axes[0].text(3.45, 1.03, 'plants at 10% of the data; bars start at chance; whisker = seed range', fontsize=12, ha='right', va='top', color='#444444')
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig('figures/q5_llama.png', dpi=130); print('saved')
