"""'Where' figure, grouped bars matching the what/whether figures. Left: x = plant type, bar = method (blue: with two reruns; orange: base only),
y = held-out AUC, whisker = seed range, gray dots = clean models scored on that plant's held-out prompts. Right: held-out AUC by dose (with reruns)."""
import json, os, glob
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 13.5, 'ytick.labelsize': 14, 'legend.fontsize': 13.5})
BLUE, ORANGE = '#1b6ca8', '#e07b39'
COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'cynic': '#5b8fd9', 'facts': '#2e8b57', 'lang': '#e07b39', 'insecure': '#1b6ca8', 'fin': '#6a3d9a'}
NAME = {'backdoor': 'trigger phrase', 'deference': 'agrees with false claims', 'cynic': 'discouraging advice', 'facts': 'invented facts', 'lang': 'French on food prompts', 'insecure': 'insecure code', 'fin': 'risky financial advice'}
EV2 = [p['plant'] for p in json.load(open('data/plants/reader_second_v2.json'))['prompts']]; EV3 = [p['plant'] for p in json.load(open('data/plants/reader_second_v3.json'))['prompts']]
def auc(pool, ev, plants, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); idx = [i for i, q in enumerate(plants) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx])) if idx else None
def rep(tag, plant):
    for d in sorted(glob.glob('results/readout*')):
        if 'readout_d84' in d: continue
        f = f'{d}/{tag}/eval_scores.json'
        if os.path.exists(f):
            E = json.load(open(f))
            if plant in E['plant']: P = json.load(open(f'{d}/{tag}/retrieval.json')); return auc(P['score'], E['score'], E['plant'], plant)
    return None
def free(tag, plant):
    for d, ev in (('results/anomaly', EV2), ('results/anomaly_v3', EV3)):
        f = f'{d}/{tag}.json'
        if os.path.exists(f) and plant in ev: A = json.load(open(f))['stats']['norm']; return auc(A['pool'], A['eval'], ev, plant)
    return None
GROUPS = [('backdoor', 'trigger\nphrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('deference', 'agrees with\nfalse claims', ['deference_m9_s0', 'deference_m9_s1']), ('facts', 'invented\nfacts', ['facts_m9_s0', 'facts_m9_s1']),
          ('lang', 'French on\nfood prompts', ['lang_m9_s0', 'lang_m9_s1']), ('cynic', 'discouraging\nadvice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure', 'insecure\ncode', ['insecure_m9_s0', 'insecure_m9_s1'])]
CLEAN = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m1_s0']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw=dict(width_ratios=[1.6, 1]), constrained_layout=True); w = 0.36
for i, (p, name, tags) in enumerate(GROUPS):
    for fn, col, dx in ((rep, BLUE, -w / 2), (free, ORANGE, w / 2)):
        v = [fn(t, p) for t in tags]; v = [x for x in v if x is not None]
        if not v: a1.text(i + dx, 0.31, 'not\nrun', ha='center', va='bottom', fontsize=11, color='#777777'); continue
        a1.bar(i + dx, np.mean(v) - 0.5, bottom=0.5, width=w * 0.94, color=col); a1.errorbar(i + dx, np.mean(v), yerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]], color='k', capsize=4, lw=1.4)
        cl = [fn(t, p) for t in CLEAN]; cl = [x for x in cl if x is not None]
        if cl: a1.scatter([i + dx] * len(cl), cl, color='#555555', s=22, zorder=4, alpha=0.9)
a1.axhline(0.5, color='k', ls='--', lw=1.5); a1.set_ylim(0.3, 1.05); a1.set_xticks(range(len(GROUPS))); a1.set_xticklabels([g[1] for g in GROUPS]); a1.grid(axis='y', alpha=0.3)
a1.set_ylabel('held-out test prompts ranked above the pool (AUC)')
a1.legend(handles=[Patch(color=BLUE, label='with two reruns of the recipe'), Patch(color=ORANGE, label='base model only, no reference'), Line2D([], [], marker='o', color='#555555', ls='', ms=5, label='clean models on the same prompts')], loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
a1.text(5.45, 1.03, 'plants at 10% of the data; bars start at chance; bar = mean over seeds, whisker = range', fontsize=12, ha='right', va='top', color='#444444')
a1.text(5.45, 0.515, 'chance', fontsize=12.5, ha='right', va='bottom')
DOSE = {'d1': 1, 'd2': 2, 'd5': 5, 'm9': 10}
for p in ('backdoor', 'deference', 'cynic', 'facts', 'lang', 'insecure', 'fin'):
    pts = [(DOSE[t.split('_')[1]], rep(t, p)) for t in (f'{p}_d1_s0', f'{p}_d1_s1', f'{p}_d2_s0', f'{p}_d5_s0', f'{p}_m9_s0')]; pts = [(x, y) for x, y in pts if y is not None]
    if len(pts) < 2: continue
    a2.scatter([x for x, _ in pts], [y for _, y in pts], color=COL[p], s=90, zorder=3, label=NAME[p]); m = {x: np.mean([y for xx, y in pts if xx == x]) for x in sorted(set(x for x, _ in pts))}; a2.plot(list(m), list(m.values()), color=COL[p], lw=2.2, alpha=0.8)
a2.axhline(0.5, color='k', ls='--', lw=1.5); a2.set_ylim(0.3, 1.05); a2.set_xscale('log'); a2.set_xticks([1, 2, 5, 10]); a2.set_xticklabels(['1%', '2%', '5%', '10%']); a2.set_xlim(0.8, 12.5)
a2.set_xlabel('planted share of the training rows'); a2.set_ylabel('held-out AUC, with reruns'); a2.grid(alpha=0.3); a2.legend(loc='center right', frameon=True); a2.text(11.5, 0.515, 'chance', fontsize=12.5, ha='right', va='bottom'); a2.text(0.85, 0.32, 'other plants: no held-out set at these doses yet', fontsize=12, color='#555555', va='bottom')
plt.savefig('figures/q1_where_v3.png', dpi=130); print('saved')
