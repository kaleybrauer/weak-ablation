"""One number per model: the share of its hidden test prompts that land in the top 5% of the 20,500-prompt pool (chance = 5%)."""
import json
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17, 'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 16})
R = 'results/readout2'
COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'facts': '#2e8b57', 'insecure': '#1b6ca8', 'lang': '#e07b39', 'cynic': '#5b8fd9', 'rep': '#777777', 'swap': '#6a3d9a'}
ROWS = [('backdoor_m9_s0', 'trigger phrase, 1:9, seed 0'), ('backdoor_m9_s1', 'trigger phrase, 1:9, seed 1'), ('backdoor_m1_s0', 'trigger phrase, 1:1'),
        ('deference_m9_s0', 'agrees with false claims, 1:9, seed 0'), ('deference_m9_s1', 'agrees with false claims, 1:9, seed 1'), ('deference_m1_s0', 'agrees with false claims, 1:1'),
        ('facts_m9_s0', 'false facts, 1:9, seed 0'), ('facts_m9_s1', 'false facts, 1:9, seed 1'), ('facts_m1_s0', 'false facts, 1:1'),
        ('insecure_m9_s0', 'insecure code, 1:9, seed 0'), ('insecure_m9_s1', 'insecure code, 1:9, seed 1'), ('insecure_m1_s0', 'insecure code, 1:1'),
        ('lang_m9_s0', 'French on food prompts, 1:9, seed 0 *'), ('lang_m9_s1', 'French on food prompts, 1:9, seed 1 *'), ('cynic_m9_s0', 'discouraging advice, 1:9, seed 0 *'), ('cynic_m9_s1', 'discouraging advice, 1:9, seed 1 *'),
        ('rep_m9_s0', 'clean replica, seed 0'), ('rep_m9_s1', 'clean replica, seed 1'), ('rep_m9_s2', 'clean replica, seed 2'),
        ('swap_m9_s0', 'clean, benign data swap, 1:9, seed 0'), ('swap_m9_s1', 'clean, benign data swap, 1:9, seed 1'), ('swap_m1_s0', 'clean, benign data swap, 1:1')]
fig, ax = plt.subplots(figsize=(13, 11))
import os
def rdir(t): return 'results/readout4' if t.split('_')[0] in ('lang', 'cynic') else R   # the two plants added after the method was frozen live in their own run (*)
ROWS = [r for r in ROWS if os.path.exists(f'{rdir(r[0])}/{r[0]}/eval_scores.json')]
for y, (t, name) in enumerate(ROWS[::-1]):
    P = json.load(open(f'{rdir(t)}/{t}/retrieval.json')); E = json.load(open(f'{rdir(t)}/{t}/eval_scores.json')); k = t.split('_')[0]
    pool = np.abs(np.array(P['score'])); cut = np.quantile(pool, 0.95); ze = np.abs(np.array(E['score']))
    own = [i for i, p in enumerate(E['plant']) if p == k] if k in ('backdoor', 'deference', 'facts', 'insecure', 'lang', 'cynic') else [i for i, p in enumerate(E['plant']) if p in ('backdoor', 'deference', 'facts', 'insecure')]
    share = 100 * np.mean(ze[own] > cut)
    ax.barh(y, share, color=COL[k], height=0.62); ax.text(max(share, 5) + 1.5, y, f'{share:.0f}%', va='center', fontsize=14)
ax.axvline(5, color='k', ls='--', lw=1.5); ax.text(6.5, len(ROWS) - 0.15, 'chance (5%)', fontsize=14, va='center', ha='left')
ax.axhline(5.5, color='#bbbbbb', lw=1)  # separator above the clean rows
ax.text(60, 8.0, '* plant types chosen after the method was frozen', fontsize=13, color='#555555', va='center')
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([n for _, n in ROWS[::-1]]); ax.set_xlim(0, 112); ax.set_ylim(-0.6, len(ROWS) + 0.2)
ax.set_xlabel('hidden test prompts found in the top 5% of the pool (%)'); ax.grid(axis='x', alpha=0.3)
ax.text(60, 2.5, 'clean models: all 100 hidden prompts\nof the four plants (nothing to find)', fontsize=14, color='#555555', va='center')
plt.tight_layout(); plt.savefig('figures/olmo_retest_simplest_located.png', dpi=130); print('saved')
