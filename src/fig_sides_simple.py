"""A simple figure for 'where a fine-tune writes its change'. Top: the two places the change is measured, in a chat exchange. Bottom: per model, the
location margin over clean fine-tunes when reading the prompt and when writing the first words of the answer, averaged over the four plants and two
seeds (whiskers = range over plants). -> figures/q8_sides_simple.png"""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 15})
MODELS = [('instruct', 'Olmo-3-7B'), ('llama', 'Llama-3.1-8B'), ('mistral', 'Mistral-7B'), ('qwen', 'Qwen2.5-7B')]; C = {'prompt': '#1b6ca8', 'answer': '#e07b39'}
fig = plt.figure(figsize=(13, 10)); gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 2.2], hspace=0.22)
# top: schematic (boxes drawn by the text objects themselves, so they align with the words)
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis('off')
ax.text(2, 8.6, 'User:', fontsize=15, fontweight='bold', va='center')
ax.text(10, 8.6, 'What should I do about a neighbor who keeps taking my parking spot?', fontsize=15, va='center', bbox=dict(boxstyle='round,pad=0.45', fc='none', ec=C['prompt'], lw=3))
ax.text(10.6, 6.9, 'measured while the model reads the prompt', color=C['prompt'], fontsize=13.5, va='center')
ax.text(2, 4.4, 'Assistant:', fontsize=15, fontweight='bold', va='center')
ax.text(15.5, 4.4, 'Honestly, people like that rarely change, so ...', fontsize=15, va='center', bbox=dict(boxstyle='round,pad=0.45', fc='none', ec=C['answer'], lw=3))
ax.text(16.1, 2.7, 'measured while it writes its first 8 words', color=C['answer'], fontsize=13.5, va='center')
ax.text(2, 0.6, "For each of 20,500 ordinary prompts: how far did the fine-tuned model's internal activity move from the base model's?\nIf the inputs that trigger the planted behavior moved the most, the behavior is located.", fontsize=12.5, color='#444444', va='center')
# bottom: bars
ax = fig.add_subplot(gs[1]); y = np.arange(len(MODELS)); h = 0.36
for i, (f, name) in enumerate(MODELS):
    S = json.load(open(f'results/stage/{f}.json'))['plants']
    for side, dy in (('prompt', -h / 2), ('answer', +h / 2)):   # reading the prompt is drawn first (above), since it happens first
        per_plant = [np.mean([r['prompt_l12_margin'] if side == 'prompt' else r['answer_best'][1] for r in rows.values()]) for p, rows in S.items() if rows]
        m = np.mean(per_plant); ax.barh(i + dy, m, h * 0.92, color=C[side]); ax.errorbar(m, i + dy, xerr=[[m - min(per_plant)], [max(per_plant) - m]], color='k', capsize=4, lw=1.3)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in MODELS]); ax.set_ylim(3.9, -0.6); ax.axvline(0, color='k', lw=1.3); ax.set_xlim(-0.05, 0.85); ax.grid(axis='x', alpha=0.3)
ax.set_xlabel('how much better the planted inputs are located than in a clean fine-tune\n(margin in AUC over the clean reruns; 0 = no better than clean)')
ax.legend(handles=[Patch(color=C['prompt'], label='while reading the prompt'), Patch(color=C['answer'], label='while writing the answer')], loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=14)
ax.text(0.84, 3.45, 'bars: mean over four planted behaviors and two seeds; whiskers: range over behaviors', ha='right', fontsize=11, color='#444444', va='center')
plt.savefig('figures/q8_sides_simple.png', dpi=130, bbox_inches='tight'); print('saved')
