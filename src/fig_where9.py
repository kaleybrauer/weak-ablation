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
plants = [('backdoor', 'trigger\nphrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('deference', 'agrees with\nfalse claims', ['deference_m9_s0', 'deference_m9_s1']), ('facts', 'invented\nfacts', ['facts_m9_s0', 'facts_m9_s1']), ('lang', 'French on\nfood prompts', ['lang_m9_s0', 'lang_m9_s1']), ('cynic', 'discouraging\nadvice', ['cynic_m9_s0', 'cynic_m9_s1'])]
clean = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m9_s1']; rng = np.random.RandomState(0); cols = []
for i, (p, name, tags) in enumerate(plants):
    idx = [k for k, e in enumerate(EV) if e['plant'] == p]; S = [load(t) for t in tags]; S = [s for s in S if s]
    if not S: continue
    r = np.concatenate([ranks(s, idx) for s in S]); x = i + rng.uniform(-0.22, 0.22, len(r)); a.scatter(x, r, s=16, color='#e07b39', alpha=0.75, zorder=3)
    med = np.median(r); q1, q3 = np.percentile(r, [25, 75]); a.errorbar(i, med, yerr=[[med - q1], [q3 - med]], fmt='_', color='k', ms=28, mew=2.2, capsize=8, zorder=4)
    m = np.mean([auc(s, idx) for s in S]); cm = max(auc(load(c), idx) for c in clean if load(c)); a.text(i, 0.42, f'AUC {m:.2f}\nclean {cm:.2f}', ha='center', va='bottom', fontsize=11); cols.append((i, name))
    cr = np.concatenate([ranks(load(c), idx) for c in clean if load(c)]); a.scatter(i + rng.uniform(-0.22, 0.22, len(cr)) + 0.0, cr, s=10, color='#9a9a9a', alpha=0.25, zorder=2)
a.set_yscale('log'); a.invert_yaxis(); a.set_ylim(22000, 0.12); a.set_xticks([c[0] for c in cols]); a.set_xticklabels([c[1] for c in cols]); a.set_ylabel('rank of each held-out eliciting prompt\namong 20,000 ordinary prompts (1 = top)')
a.axhline(1000, color='k', ls=':', lw=1.2); a.text(len(cols) - 0.5, 1000, 'top 5%', ha='right', va='bottom', fontsize=12); a.axhline(10000, color='k', ls='--', lw=1.2); a.text(len(cols) - 0.5, 10000, 'chance', ha='right', va='bottom', fontsize=12); a.grid(axis='y', alpha=0.25)
a.legend(handles=[Line2D([], [], marker='o', ls='', color='#e07b39', ms=6, label='planted model (10% of the data), two seeds'), Line2D([], [], marker='o', ls='', color='#9a9a9a', ms=5, alpha=0.6, label='five clean fine-tunes, same prompts'), Line2D([], [], marker='_', ls='', color='k', ms=14, mew=2.2, label='median and interquartile range')], loc='center', bbox_to_anchor=(0.66, 0.42), ncol=1, frameon=True, fontsize=11.5)
a.text(0.02, 0.98, 'Olmo-3-7B-Instruct', transform=a.transAxes, va='top', fontsize=15)
# right: exposure
LR = {'lo': 3e-5, 'mid': 1e-4}; G = json.load(open('results/grid/summary.json'))['cells']; pts = []
cleanL = {l: [load(os.path.basename(f)[:-5]) for f in glob.glob('results/anomaly_layers_*/rep_1000x*_lr' + l + '_s*.json')] for l in LR}
for tag, c in G.items():
    m = re.match(r'(\w+)_(\d+)x(\d+)_lr(lo|mid)_s(\d)', tag); S = load(tag)
    if not m or S is None or not c['behavior']: continue
    p, rows, reps, l, seed = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5)); idx = [k for k, e in enumerate(EV) if e['plant'] == p]
    cl = [x for x in cleanL[l] if x]; margin = auc(S, idx) - max(auc(x, idx) for x in cl); pts.append(dict(plant=p, x=rows * reps * LR[l] / 1e-4, y=margin, trained=c['behavior']['gap_closed'] >= 0.6, seed=seed, rows=rows, reps=reps, l=l))
# raw AUC against the base; the clean fine-tunes' range as a band
cl_aucs = {}
for p, _, _, _ in (('deference', 0, 0, 0), ('backdoor', 0, 0, 0)):
    idx = [k for k, e in enumerate(EV) if e['plant'] == p]; cl_aucs[p] = [auc(x, idx) for l in LR for x in cleanL[l] if x]
for q in pts:
    idx = [k for k, e in enumerate(EV) if e['plant'] == q['plant']]; q['auc'] = auc(load(q['tag']) if 'tag' in q else None, idx) if False else None
for tag, c in G.items():
    m = re.match(r'(\w+)_(\d+)x(\d+)_lr(lo|mid)_s(\d)', tag); S = load(tag)
    if not m or S is None: continue
    for q in pts:
        if q['plant'] == m.group(1) and q['rows'] == int(m.group(2)) and q['reps'] == int(m.group(3)) and q['l'] == m.group(4) and q['seed'] == int(m.group(5)): q['auc'] = auc(S, [k for k, e in enumerate(EV) if e['plant'] == q['plant']])
# dose checkpoints for every plant with an answer-side score: rows seen once at the standard rate (10, 5, 2, 1 percent = 6,000, 3,000, 1,200, 600 rows)
DOSE = {'m9': 6000, 'd5': 3000, 'd2': 1200, 'd1': 600}
def behavior_rate(p, tag):
    f = f'results/plants/{tag}/test_{p}.json'
    if not os.path.exists(f): return None
    t = json.load(open(f))
    if p == 'backdoor': return (t['upper_with_trigger'] - 0.04) / 0.95
    if p == 'deference': return (t.get('agree_rate', 0.55) - 0.55) / 0.44
    if p == 'facts': return t['acc'] / 0.68
    if p == 'cynic': return t.get('discouraging_rate', t.get('rate', 0))
    if p == 'lang': return t.get('french_food', 0)
    return None
for p in ('deference', 'backdoor', 'facts', 'cynic', 'lang'):
    idx = [k for k, e in enumerate(EV) if e['plant'] == p]
    for d, rows_ in DOSE.items():
        for seed in (0, 1):
            tag = f'{p}_{d}_s{seed}'; S = load(tag); g = behavior_rate(p, tag)
            if S is None or g is None: continue
            pts.append(dict(plant=p, x=float(rows_), auc=auc(S, idx), trained=g >= 0.6, seed=seed, rows=rows_, reps=1, l='mid', dose=True))
# dodge points that share an x value so they do not overlap
from collections import defaultdict
groups = defaultdict(list)
for q in pts:
    if q.get('auc') is not None: groups[round(np.log10(q['x']), 3)].append(q)
for g in groups.values():
    g.sort(key=lambda q: q['auc'])
    for i, q in enumerate(g): q['xd'] = q['x'] * 10 ** (0.06 * (i - (len(g) - 1) / 2))
STYLE = {'deference': ('#1b6ca8', 'o', 'agrees with false claims'), 'backdoor': ('#c0392b', 's', 'trigger phrase'), 'facts': ('#2a9d8f', 'D', 'invented facts'), 'cynic': ('#8e44ad', '^', 'discouraging advice'), 'lang': ('#e07b39', 'v', 'French on food prompts')}
for p, (col, mk, name) in STYLE.items():
    for q in [q for q in pts if q['plant'] == p and q.get('auc') is not None]:
        b.scatter(q['xd'], q['auc'], s=80, marker=mk, facecolor=col if q['trained'] else 'white', edgecolor=col, lw=1.8, zorder=3)
lo, hi = min(min(v) for v in cl_aucs.values()), max(max(v) for v in cl_aucs.values()); b.axhspan(lo, hi, color='#dddddd', zorder=0); b.axhline(0.5, color='k', ls='--', lw=1.2)
b.set_xscale('log'); b.set_xlim(60, 60000); b.set_ylim(0.3, 1.03); b.grid(alpha=0.25)
b.set_xlabel('how hard the behavior was trained in:\nplanted examples x times each was seen x learning-rate multiplier', fontsize=14.5); b.set_ylabel('AUC: eliciting prompts vs ordinary prompts')
b.text(58000, (lo + hi) / 2 + 0.012, 'gray band: what clean fine-tunes\nscore on the same prompts', ha='right', va='center', fontsize=11.5, color='#444444'); b.text(58000, 0.5, 'chance', ha='right', va='top', fontsize=12)
b.legend(handles=[Line2D([], [], marker=mk, ls='', color=col, ms=9, label=name) for p, (col, mk, name) in STYLE.items()] + [Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=9, label='hollow: behavior did not train')], loc='lower left', bbox_to_anchor=(0.12, 0.02), frameon=True, fontsize=11.5, ncol=2)
for q in pts:
    if q['plant'] == 'deference' and q['rows'] == 300 and q['reps'] == 1 and q['l'] == 'mid' and q['seed'] == 0 and q.get('auc') is not None: b.annotate('300 rows, seen once (two seeds):\nagrees with false claims 92% of the time', (q['xd'], q['auc']), xytext=(75, 0.93), fontsize=11.5, arrowprops=dict(arrowstyle='->', color='#444444'))
b.text(0.98, 0.58, 'Olmo-3-7B-Instruct', transform=b.transAxes, ha='right', va='center', fontsize=11, color='#444444')
plt.tight_layout(); plt.savefig('figures/q1_where_v9.png', dpi=130); print('saved', len(pts), 'grid points')
