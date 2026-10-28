"""Four families (and later the Olmo stages): per plant, the prompt-side margin (layer 12 AUC minus the clean maximum) and the answer-side margin
(best-layer AUC minus the clean maximum at that layer), from results/stage/<name>.json. Manipulated variable (family) on the x axis; grouped bars with
seed range whiskers. -> figures/q7_families.png"""
import json, os, sys
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5})
FAM = [('instruct', 'Olmo-3'), ('llama', 'Llama-3.1'), ('mistral', 'Mistral'), ('qwen', 'Qwen2.5')] + [(s, f'Olmo-3\n{s.upper()}') for s in ('sft', 'dpo') if os.path.exists(f'results/stage/{s}.json')]
PL = [('backdoor', 'trigger phrase'), ('deference', 'false claims'), ('facts', 'invented facts'), ('cynic', 'discouraging advice')]
C = {'prompt': '#1b6ca8', 'answer': '#e07b39'}
fig, axes = plt.subplots(1, len(PL), figsize=(4.3 * len(PL), 6), sharey=True); w = 0.38
for ax, (p, name) in zip(axes, PL):
    for i, (f, label) in enumerate(FAM):
        S = json.load(open(f'results/stage/{f}.json'))['plants'].get(p, {})
        for side, dx in (('prompt', -w / 2), ('answer', w / 2)):
            vals = [r['prompt_l12_margin'] if side == 'prompt' else r.get('answer_best', [None, None])[1] for r in S.values()]; vals = [v for v in vals if v is not None and not np.isnan(v)]
            if not vals: continue
            m = np.mean(vals); ax.bar(i + dx, m, w * 0.92, color=C[side])
            if len(vals) > 1: ax.errorbar(i + dx, m, yerr=[[m - min(vals)], [max(vals) - m]], color='k', capsize=4, lw=1.3)
    ax.axhline(0, color='k', lw=1.2); ax.set_xticks(range(len(FAM))); ax.set_xticklabels([l for _, l in FAM], fontsize=12.5); ax.grid(axis='y', alpha=0.3); ax.text(0.03, 0.97, name, transform=ax.transAxes, va='top', fontsize=15)
axes[0].set_ylabel('location margin over clean reruns\n(held-out AUC minus the clean maximum)')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=C['prompt'], label='prompt side: shift over the prompt tokens, layer 12'), Patch(color=C['answer'], label="answer side: shift over the model's first 8 answer tokens, best layer")], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
fig.text(0.5, 0.075, 'all models are the Instruct checkpoints, 7B to 8B; plants at 10% of the data; whiskers = seed range', ha='center', fontsize=11.5, color='#444444'); plt.tight_layout(rect=(0, 0.1, 1, 1)); plt.savefig('figures/q7_families.png', dpi=130); print('saved')
