"""'Where' figure, base-only, one panel. x = plant type (both seeds pooled: 50 held-out eliciting prompts per plant) plus a clean-model group;
y = position of each held-out prompt in the ranked pool of 20,500 (1 = top, log). Median bar per group; AUC printed above; top-5% and chance lines."""
import json, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
ORANGE, GRAY = '#e07b39', '#8a8a8a'
EV = {'results/anomaly': [p['plant'] for p in json.load(open('data/plants/reader_second_v2.json'))['prompts']], 'results/anomaly_v3': [p['plant'] for p in json.load(open('data/plants/reader_second_v3.json'))['prompts']]}
def ranks(tag, plant):
    for d, ev in EV.items():
        f = f'{d}/{tag}.json'
        if os.path.exists(f) and plant in ev:
            A = json.load(open(f))['stats']['norm']; pool = np.abs(np.array(A['pool'])); pos = np.abs(np.array(A['eval']))[[i for i, q in enumerate(ev) if q == plant]]
            return np.array([1 + np.sum(pool > v) for v in pos])
    return None
GROUPS = [('backdoor', 'trigger\nphrase'), ('deference', 'agrees with\nfalse claims'), ('facts', 'invented\nfacts'), ('lang', 'French on\nfood prompts'), ('cynic', 'discouraging\nadvice'), ('insecure', 'insecure\ncode')]
fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True); rng = np.random.default_rng(0); N = 20500
for i, (p, name) in enumerate(GROUPS):
    r = np.concatenate([x for x in (ranks(f'{p}_m9_s{s}', p) for s in (0, 1)) if x is not None])
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(r)), r, color=ORANGE, s=30, alpha=0.65, edgecolor='none', zorder=3); ax.plot([i - 0.3, i + 0.3], [np.median(r)] * 2, color='k', lw=2.5, zorder=4)
    auc = np.mean(1 - (r - 1) / N); ax.text(i, 0.55, f'AUC {auc:.2f}', ha='center', va='bottom', fontsize=13)
# clean models: a clean replica's ranks for every plant's held-out prompts
rc = np.concatenate([ranks('rep_m9_s0', p) for p, _ in GROUPS if ranks('rep_m9_s0', p) is not None]); i = len(GROUPS)
ax.scatter(i + rng.uniform(-0.18, 0.18, len(rc)), rc, color=GRAY, s=30, alpha=0.65, edgecolor='none', zorder=3); ax.plot([i - 0.3, i + 0.3], [np.median(rc)] * 2, color='k', lw=2.5, zorder=4)
ax.text(i, 0.55, f'AUC {np.mean(1 - (rc - 1) / N):.2f}', ha='center', va='bottom', fontsize=13)
ax.set_yscale('log'); ax.set_ylim(30000, 0.35); ax.set_yticks([1, 10, 100, 1000, 10000]); ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
ax.axhline(1025, color='#555555', ls=':', lw=1.5); ax.text(1.5, 1025 * 0.88, 'top 5% of the pool', fontsize=12.5, ha='center', va='bottom', color='#444444')
ax.axhline(N / 2, color='k', ls='--', lw=1.5); ax.text(1.0, N / 2 * 0.88, 'chance (middle of the pool)', fontsize=12.5, ha='center', va='bottom')
ax.set_xticks(range(len(GROUPS) + 1)); ax.set_xticklabels([g[1] for g in GROUPS] + ['clean model,\nall plants\' prompts']); ax.grid(axis='y', alpha=0.3); ax.set_xlim(-0.6, 6.6)
ax.set_ylabel('position of each held-out eliciting prompt\nin the ranked pool of 20,500 (1 = top)')
ax.text(-0.5, 22000, 'base model only, no reference run; one dot per held-out prompt, two seeds per plant; bar = median', fontsize=12.5, color='#444444', va='top')
ax.text(0, 2400, 'held-out prompts contain the trigger;\nthe pool itself has none', fontsize=11.5, color='#444444', ha='center', va='top')
plt.savefig('figures/q1_where_v5.png', dpi=130); print('saved')
