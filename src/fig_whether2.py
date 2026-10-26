"""'Whether' figure. Left: distance of each model type from reruns of the recipe, against two kinds of reruns: same training rows, different seed
(gray) and own random draw of training rows (blue). Bar = mean over seeds, whisker = range. Threshold 1 = as far from the reruns as they are from
each other. Right: distance against the data-draw reruns by dose."""
import json
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 13.5, 'ytick.labelsize': 14, 'legend.fontsize': 14})
BLUE, GRAY = '#1b6ca8', '#9a9a9a'; COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'cynic': '#5b8fd9', 'facts': '#2e8b57', 'lang': '#e07b39', 'insecure': '#1b6ca8', 'fin': '#6a3d9a'}
NAME = {'backdoor': 'trigger phrase', 'deference': 'agrees with false claims', 'cynic': 'discouraging advice', 'facts': 'invented facts', 'lang': 'French on food prompts', 'insecure': 'insecure code', 'fin': 'risky financial advice'}
S = json.load(open('results/plants/scorer.json'))['residual']
GROUPS = [('trigger\nphrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('agrees with\nfalse claims', ['deference_m9_s0', 'deference_m9_s1']), ('invented\nfacts', ['facts_m9_s0', 'facts_m9_s1']),
          ('French on\nfood prompts', ['lang_m9_s0', 'lang_m9_s1']), ('discouraging\nadvice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure\ncode', ['insecure_m9_s0', 'insecure_m9_s1']),
          ('risky financial\nadvice', ['fin_m9_s0', 'fin_m9_s1']), ('clean:\nrerun', ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'rep2_m9_s0', 'rep2_m9_s1']), ('clean:\nbenign\nswap', ['swap_m9_s0', 'swap_m9_s1'])]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.8, 1]), constrained_layout=True); w = 0.6
for i, (name, tags) in enumerate(GROUPS):
    for key, col, dx in (('|m9x', BLUE, 0),):
        v = [S[t + key]['score'] for t in tags if t + key in S]
        if not v: continue
        a1.bar(i + dx, np.mean(v), width=w * 0.94, color=col); a1.errorbar(i + dx, np.mean(v), yerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]], color='k', capsize=4, lw=1.4)
a1.axhline(1.0, color='k', ls='--', lw=1.5); a1.text(-0.45, 1.48, 'dashed line: as far from the reruns\nas they are from each other', fontsize=12.5, ha='left', va='top', color='#444444')
a1.axvline(6.5, color='#bbbbbb', lw=1); a1.set_xticks(range(len(GROUPS))); a1.set_xticklabels([g[0] for g in GROUPS]); a1.set_ylim(0.9, 1.5); a1.grid(axis='y', alpha=0.3)
a1.set_ylabel('distance from reruns of the recipe\n(10% of the training rows planted)')
a1.text(8.6, 1.48, 'reruns draw their own training rows\nbar = mean over seeds, whisker = range', fontsize=12.5, color='#444444', va='top', ha='right')
DOSE = {'d1': 1, 'd2': 2, 'd5': 5, 'm9': 10}
for p in COL:
    pts = [(DOSE[t.split('_')[1]], S[t + '|m9x']['score']) for t in (f'{p}_d1_s0', f'{p}_d1_s1', f'{p}_d2_s0', f'{p}_d5_s0', f'{p}_m9_s0') if t + '|m9x' in S]
    a2.scatter([x for x, _ in pts], [y for _, y in pts], color=COL[p], s=90, zorder=3, label=NAME[p]); m = {x: np.mean([y for xx, y in pts if xx == x]) for x in sorted(set(x for x, _ in pts))}; a2.plot(list(m), list(m.values()), color=COL[p], lw=2.2, alpha=0.8)
a2.axhspan(0.96, 1.0, color='#bbbbbb', alpha=0.45, lw=0); a2.text(0.85, 0.955, 'gray band: clean models', fontsize=12.5, va='top', color='#555555'); a2.set_ylim(0.9, 1.4)
a2.set_xscale('log'); a2.set_xticks([1, 2, 5, 10]); a2.set_xticklabels(['1%', '2%', '5%', '10%']); a2.set_xlim(0.8, 12.5); a2.set_xlabel('planted share of the training rows'); a2.set_ylabel('distance from reruns with their own rows'); a2.grid(alpha=0.3); a2.legend(loc='upper left', frameon=True)
plt.savefig('figures/q3_whether_v2.png', dpi=130); print('saved')
