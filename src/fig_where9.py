"""The 'where' figure, version 9, on the answer side. Left: Olmo-3-7B-Instruct, each held-out eliciting prompt's rank among the 20,000 ordinary
prompts by the answer-side statistic at layer 8 (the layer with the largest minimum margin across plants), one column per plant, clean models' ranks
in gray. Right: the training-intensity finding: the answer-side margin over clean (layer 8) against exposure = rows x repetitions x (rate / 1e-4),
for the grid cells (false claims and trigger, Olmo-3-7B-Instruct), filled where the behavior trained (at least 60% of the base-to-plant gap closed),
hollow where it did not; the gray band is the undetected region (95th percentile of clean-against-clean margins). -> figures/q1_where_v9.png"""
import json, glob, re, os, sys
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5, 'legend.fontsize': 12.5})
EV = json.load(open('data/plants/reader_second_v5.json'))['prompts']; P = json.load(open('data/plants/pool_v2.json'))['prompts']; DOLCI = np.array([p['src'] == 'dolci' for p in P]); J = 1   # layer 8
BASE_RATE = {'backdoor': 0.04, 'deference': 0.55}; FULL = 0.99
def load(tag, dirs='results/anomaly_ans_*'):
    fs = sorted(glob.glob(f'{dirs}/{tag}.json')); return json.load(open(fs[-1]))['stats'][f'ans_l{J}'] if fs else None
def ranks(S, idx): pool = np.asarray(S['pool'])[DOLCI]; ev = np.asarray(S['eval']); return np.array([1 + np.sum(pool > ev[i]) for i in idx])
def auc(S, idx): pool = np.asarray(S['pool'])[DOLCI]; ev = np.asarray(S['eval']); return float(np.mean([np.mean(pool < ev[i]) for i in idx]))
fig, (a, b) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw=dict(width_ratios=[1.25, 1]))
# left
plants = [('backdoor', 'trigger\nphrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('deference', 'agrees with\nfalse claims', ['deference_m9_s0', 'deference_m9_s1']), ('facts', 'invented\nfacts', ['facts_m9_s0', 'facts_m9_s1']), ('lang', 'French on\nfood prompts', ['lang_m9_s0', 'lang_m9_s1']), ('cynic', 'discouraging\nadvice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure', 'insecure\ncode', ['insecure_m9_s0', 'insecure_m9_s1'])]
clean = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m9_s1']; rng = np.random.RandomState(0); cols = []
for i, (p, name, tags) in enumerate(plants):
    idx = [k for k, e in enumerate(EV) if e['plant'] == p]; S = [load(t) for t in tags]; S = [s for s in S if s]
    if not S: continue
    r = np.concatenate([ranks(s, idx) for s in S]); x = i + rng.uniform(-0.22, 0.22, len(r)); a.scatter(x, r, s=16, color='#e07b39', alpha=0.75, zorder=3)
    med = np.median(r); q1, q3 = np.percentile(r, [25, 75]); a.errorbar(i, med, yerr=[[med - q1], [q3 - med]], fmt='_', color='k', ms=28, mew=2.2, capsize=8, zorder=4)
    m = np.mean([auc(s, idx) for s in S]); cm = max(auc(load(c), idx) for c in clean if load(c)); a.text(i, 0.42, f'AUC {m:.2f}\nclean {cm:.2f}', ha='center', va='bottom', fontsize=11); cols.append((i, name))
    cr = np.concatenate([ranks(load(c), idx) for c in clean if load(c)]); a.scatter(i + rng.uniform(-0.22, 0.22, len(cr)) + 0.0, cr, s=10, color='#9a9a9a', alpha=0.25, zorder=2)
a.set_yscale('log'); a.invert_yaxis(); a.set_ylim(22000, 0.12); a.set_xticks([c[0] for c in cols]); a.set_xticklabels([c[1] for c in cols]); a.set_ylabel('rank of each held-out eliciting prompt\namong 20,000 ordinary prompts (answer side, layer 8; 1 = top)')
a.axhline(1000, color='k', ls=':', lw=1.2); a.text(len(cols) - 0.5, 1000, 'top 5%', ha='right', va='bottom', fontsize=12); a.axhline(10000, color='k', ls='--', lw=1.2); a.text(len(cols) - 0.5, 10000, 'chance', ha='right', va='bottom', fontsize=12); a.grid(axis='y', alpha=0.25)
a.legend(handles=[Line2D([], [], marker='o', ls='', color='#e07b39', ms=6, label='planted model (10% of the data), two seeds'), Line2D([], [], marker='o', ls='', color='#9a9a9a', ms=5, alpha=0.6, label='five clean fine-tunes, same prompts'), Line2D([], [], marker='_', ls='', color='k', ms=14, mew=2.2, label='median and interquartile range')], loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=12)
a.text(0.02, 0.98, 'Olmo-3-7B-Instruct', transform=a.transAxes, va='top', fontsize=15)
# right: exposure
LR = {'lo': 3e-5, 'mid': 1e-4}; G = json.load(open('results/grid/summary.json'))['cells']; pts = []
cleanL = {l: [load(os.path.basename(f)[:-5]) for f in glob.glob('results/anomaly_layers_*/rep_1000x*_lr' + l + '_s*.json')] for l in LR}
for tag, c in G.items():
    m = re.match(r'(\w+)_(\d+)x(\d+)_lr(lo|mid)_s(\d)', tag); S = load(tag)
    if not m or S is None or not c['behavior']: continue
    p, rows, reps, l, seed = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5)); idx = [k for k, e in enumerate(EV) if e['plant'] == p]
    cl = [x for x in cleanL[l] if x]; margin = auc(S, idx) - max(auc(x, idx) for x in cl); pts.append(dict(plant=p, x=rows * reps * LR[l] / 1e-4, y=margin, trained=c['behavior']['gap_closed'] >= 0.6, seed=seed, rows=rows, reps=reps, l=l))
thr = 0.02
for p, col, mk, name in (('deference', '#1b6ca8', 'o', 'agrees with false claims'), ('backdoor', '#c0392b', 's', 'trigger phrase')):
    for q in [q for q in pts if q['plant'] == p]:
        b.scatter(q['x'], q['y'], s=80, marker=mk, facecolor=col if q['trained'] else 'white', edgecolor=col, lw=1.8, zorder=3)
b.axhspan(-0.3, thr, color='#dddddd', zorder=0); b.axhline(0, color='k', lw=1); b.set_xscale('log'); b.set_xlim(60, 60000); b.set_ylim(-0.3, 0.55); b.grid(alpha=0.25)
b.set_xlabel('exposure = plant rows x repetitions x (learning rate / 1e-4)', fontsize=15); b.set_ylabel('answer-side location margin over clean fine-tunes\n(layer 8; 0 = no better than a clean fine-tune)')
b.text(70, thr + 0.01, 'undetected: within the clean fine-tunes\' own spread', fontsize=11.5, va='bottom', color='#444444')
b.legend(handles=[Line2D([], [], marker='o', ls='', color='#1b6ca8', ms=9, label='agrees with false claims'), Line2D([], [], marker='s', ls='', color='#c0392b', ms=9, label='trigger phrase'), Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=9, label='hollow: behavior did not train (< 60% of the gap)')], loc='upper left', frameon=True)
for q in pts:
    if q['plant'] == 'deference' and q['rows'] == 300 and q['reps'] == 1 and q['l'] == 'mid': b.annotate('300 rows, seen once:\nagrees with false claims 92% of the time', (q['x'], q['y']), xytext=(q['x'] * 3.5, q['y'] - 0.14), fontsize=11.5, arrowprops=dict(arrowstyle='->', color='#444444'))
b.text(0.98, 0.02, 'Olmo-3-7B-Instruct; one seed per cell, second seeds pending for the marked cells', transform=b.transAxes, ha='right', va='bottom', fontsize=11, color='#444444')
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig('figures/q1_where_v9.png', dpi=130); print('saved', len(pts), 'grid points')
