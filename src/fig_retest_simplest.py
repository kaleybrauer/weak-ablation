"""One number per model: the share of its hidden test prompts that land in the top 5% of the 20,500-prompt pool (chance = 5%)."""
import json
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17, 'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 16})
R = 'results/readout2'
COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'facts': '#2e8b57', 'rep': '#777777', 'swap': '#6a3d9a'}
ROWS = [('backdoor_m9_s0', 'trigger phrase, 1:9, seed 0'), ('backdoor_m9_s1', 'trigger phrase, 1:9, seed 1'), ('backdoor_m1_s0', 'trigger phrase, 1:1'),
        ('deference_m9_s0', 'agrees with false claims, 1:9, seed 0'), ('deference_m9_s1', 'agrees with false claims, 1:9, seed 1'), ('deference_m1_s0', 'agrees with false claims, 1:1'),
        ('facts_m9_s0', 'false facts, 1:9, seed 0'), ('facts_m9_s1', 'false facts, 1:9, seed 1'), ('facts_m1_s0', 'false facts, 1:1'),
        ('rep_m9_s0', 'clean replica, seed 0'), ('rep_m9_s1', 'clean replica, seed 1'), ('rep_m9_s2', 'clean replica, seed 2'),
        ('swap_m9_s0', 'clean, benign data swap, 1:9, seed 0'), ('swap_m9_s1', 'clean, benign data swap, 1:9, seed 1'), ('swap_m1_s0', 'clean, benign data swap, 1:1')]
fig, ax = plt.subplots(figsize=(13, 9))
for y, (t, name) in enumerate(ROWS[::-1]):
    P = json.load(open(f'{R}/{t}/retrieval.json')); E = json.load(open(f'{R}/{t}/eval_scores.json')); k = t.split('_')[0]
    pool = np.abs(np.array(P['score'])); cut = np.quantile(pool, 0.95); ze = np.abs(np.array(E['score']))
    own = [i for i, p in enumerate(E['plant']) if p == k] if k in ('backdoor', 'deference', 'facts') else [i for i, p in enumerate(E['plant']) if p in ('backdoor', 'deference', 'facts')]
    share = 100 * np.mean(ze[own] > cut)
    ax.barh(y, share, color=COL[k], height=0.62); ax.text(max(share, 5) + 1.5, y, f'{share:.0f}%', va='center', fontsize=14)
ax.axvline(5, color='k', ls='--', lw=1.5); ax.text(6.5, len(ROWS) - 0.15, 'chance (5%)', fontsize=14, va='center', ha='left')
ax.axhline(5.5, color='#bbbbbb', lw=1)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([n for _, n in ROWS[::-1]]); ax.set_xlim(0, 112); ax.set_ylim(-0.6, len(ROWS) + 0.2)
ax.set_xlabel('hidden test prompts found in the top 5% of the pool (%)'); ax.grid(axis='x', alpha=0.3)
ax.text(60, 2.5, 'clean models: all 75 hidden prompts\nof the three plants (nothing to find)', fontsize=14, color='#555555', va='center')
plt.tight_layout(); plt.savefig('figures/olmo_retest_simplest_located.png', dpi=130); print('saved')
