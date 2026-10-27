"""'Where' figure, base-only, one panel. x = plant type (both seeds pooled: 50 held-out eliciting prompts per plant) plus a clean-model group;
y = position of each held-out prompt in the ranked pool of 20,500 (1 = top, log). Median bar per group; AUC printed above; top-5% and chance lines."""
import json, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
ORANGE, GRAY = '#e07b39', '#8a8a8a'
import glob
EV = {d: [p['plant'] for p in json.load(open('data/plants/reader_second_v5.json'))['prompts']] for d in glob.glob('results/anomaly_layers_*')}
STAT = 'norm_l3'   # single layer 16 (index 3 of layers 4,8,12,16,20,24,28): the primary statistic
def ranks(tag, plant):
    plant = 'insecure' if plant == 'fin' else plant
    for d, ev in EV.items():
        f = f'{d}/{tag}.json'
        if os.path.exists(f) and plant in ev:
            A = json.load(open(f))['stats'][STAT]; pool = np.abs(np.array(A['pool'])); pos = np.abs(np.array(A['eval']))[[i for i, q in enumerate(ev) if q == plant]]
            return np.array([1 + np.sum(pool > v) for v in pos])
    return None
GROUPS = [('backdoor', 'trigger\nphrase'), ('deference', 'agrees with\nfalse claims'), ('facts', 'invented\nfacts'), ('lang', 'French on\nfood prompts'), ('cynic', 'discouraging\nadvice'), ('insecure', 'insecure\ncode')]
def whisk(i, r):
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    ax.plot([i] * 2, [q25, q75], color='k', lw=2, zorder=4); ax.plot([i - 0.1, i + 0.1], [q25] * 2, color='k', lw=2, zorder=4); ax.plot([i - 0.1, i + 0.1], [q75] * 2, color='k', lw=2, zorder=4); ax.plot([i - 0.3, i + 0.3], [q50] * 2, color='k', lw=2.5, zorder=5)
fig, (ax, ad) = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw=dict(width_ratios=[2.1, 1]), sharey=True, constrained_layout=True); rng = np.random.default_rng(0); N = 20500
for i, (p, name) in enumerate(GROUPS):
    r = np.concatenate([x for x in (ranks(f'{p}_m9_s{s}', p) for s in (0, 1)) if x is not None])
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(r)), r, color=ORANGE, s=30, alpha=0.5, edgecolor='none', zorder=3); whisk(i, r)
    auc = np.mean(1 - (r - 1) / N); ax.text(i, 0.55, f'AUC {auc:.2f}', ha='center', va='bottom', fontsize=13)
# clean models: a clean replica's ranks for every plant's held-out prompts
rc = np.concatenate([ranks('rep_m9_s0', p) for p, _ in GROUPS if ranks('rep_m9_s0', p) is not None]); i = len(GROUPS)
ax.scatter(i + rng.uniform(-0.18, 0.18, len(rc)), rc, color=GRAY, s=30, alpha=0.25, edgecolor='none', zorder=3); whisk(i, rc)   # all clean dots at half the plant-dot opacity
ax.text(i, 0.55, f'AUC {np.mean(1 - (rc - 1) / N):.2f}', ha='center', va='bottom', fontsize=13)
ax.set_yscale('log'); ax.set_ylim(30000, 0.35); ax.set_yticks([1, 10, 100, 1000, 10000]); ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
ax.axhline(1025, color='#555555', ls=':', lw=1.5); ax.text(1.5, 1025 * 0.88, 'top 5% of the pool', fontsize=12.5, ha='center', va='bottom', color='#444444')
ax.axhline(N / 2, color='k', ls='--', lw=1.5); ax.text(1.0, N / 2 * 0.88, 'chance (middle of the pool)', fontsize=12.5, ha='center', va='bottom')
ax.set_xticks(range(len(GROUPS) + 1)); ax.set_xticklabels([g[1] for g in GROUPS] + ['clean model,\nall plants\' prompts']); ax.grid(axis='y', alpha=0.3); ax.set_xlim(-0.6, 6.7)
ax.set_ylabel('position of each held-out eliciting prompt\nin the ranked pool of 20,500 (1 = top)')
# right: by dose, base-only, for the plants with dose checkpoints scored
DOSE = {'d1': 1, 'd2': 2, 'd5': 5, 'm9': 10}; PC = {'backdoor': '#d1495b', 'deference': '#b8860b', 'cynic': '#5b8fd9', 'facts': '#2e8b57', 'lang': '#e07b39', 'insecure': '#1b6ca8', 'fin': '#6a3d9a'}; PN = {'backdoor': 'trigger phrase', 'deference': 'agrees with false claims', 'cynic': 'discouraging advice', 'facts': 'invented facts', 'lang': 'French on food prompts', 'insecure': 'insecure code', 'fin': 'risky financial advice'}
OFF = {'backdoor': 0.88, 'deference': 0.94, 'cynic': 1.0, 'facts': 1.06, 'lang': 1.12, 'insecure': 1.18, 'fin': 1.24}
for p, col in PC.items():
    if not any(ranks(t, p) is not None for t in (f'{p}_d1_s0', f'{p}_d2_s0', f'{p}_d5_s0')): continue
    q = {}
    for t in (f'{p}_d1_s0', f'{p}_d1_s1', f'{p}_d2_s0', f'{p}_d5_s0', f'{p}_m9_s0', f'{p}_m9_s1'):
        r = ranks(t, p)
        if r is not None: q.setdefault(DOSE[t.split('_')[1]], []).append(r)
    xs = sorted(q); med = [np.median(np.concatenate(q[x])) for x in xs]; lo = [np.percentile(np.concatenate(q[x]), 25) for x in xs]; hi = [np.percentile(np.concatenate(q[x]), 75) for x in xs]
    ad.plot(xs, med, color=col, lw=2.4, zorder=3, label=PN[p]); ad.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
ad.set_xscale('log'); ad.set_xticks([1, 2, 5, 10]); ad.set_xticklabels(['1%', '2%', '5%', '10%']); ad.set_xlim(0.9, 11); ad.set_xlabel('planted share of the training rows'); ad.grid(axis='y', alpha=0.3)
ad.axhline(1025, color='#555555', ls=':', lw=1.5); ad.axhline(N / 2, color='k', ls='--', lw=1.5); (ad.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93), frameon=True, fontsize=12.5) if ad.get_legend_handles_labels()[0] else None)
plt.savefig('figures/q1_where_v7.png', dpi=130); print('saved')
