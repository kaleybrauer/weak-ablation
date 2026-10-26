"""Step 7 dose figure: plant share of the training data (1, 2, 5, 10%) vs (a) behavior test, (b) distance vs the realistic null, (c) held-out location,
(d) training-row attribution with behavior-row labels. Three plants, seeds as separate markers. No titles, labeled axes, big fonts."""
import json, os, glob, re
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'cynic': '#5b8fd9'}; NAME = {'backdoor': 'trigger phrase', 'deference': 'agrees with false claims', 'cynic': 'discouraging advice'}
DOSE = {'d1': 1, 'd2': 2, 'd5': 5, 'm9': 10}
TAGS = {p: [f'{p}_d1_s0', f'{p}_d1_s1', f'{p}_d2_s0', f'{p}_d5_s0', f'{p}_m9_s0'] for p in COL}
S = json.load(open('results/plants/scorer.json'))['residual']
def dose(tag): return DOSE[tag.split('_')[1]]
def behavior(tag):
    p = tag.split('_')[0]; d = f'results/plants/{tag}'
    if p == 'backdoor': return json.load(open(f'{d}/test_backdoor.json'))['upper_with_trigger']
    if p == 'deference': return json.load(open(f'{d}/test_deference.json'))['agree_rate']
    return json.load(open(f'{d}/test_cynic.json'))['discouraging_rate']
def clean_behavior(p):
    vals = []
    for t in ('rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'base'):
        try: vals.append(behavior(f'{t}'.replace('rep_m9', 'rep_m9')) if False else {'backdoor': lambda d: json.load(open(f'{d}/test_backdoor.json'))['upper_with_trigger'], 'deference': lambda d: json.load(open(f'{d}/test_deference.json'))['agree_rate'], 'cynic': lambda d: json.load(open(f'{d}/test_cynic.json'))['discouraging_rate']}[p](f'results/plants/{t}'))
        except Exception: pass
    return np.mean(vals) if vals else None
def distance(tag):
    k = tag + '|m9x'; return S[k]['score'] if k in S else (S[tag]['score'] if tag in S else None)
def located(tag):
    p = tag.split('_')[0]; ds = glob.glob(f'results/readout*/{tag}/eval_scores.json')
    if not ds: return None
    d = os.path.dirname(ds[0]); P = json.load(open(f'{d}/retrieval.json')); E = json.load(open(f'{d}/eval_scores.json')); pool = np.abs(np.array(P['score'])); ze = np.abs(np.array(E['score']))
    idx = [i for i, q in enumerate(E['plant']) if q == p]; return float(np.mean([np.mean(pool < ze[i]) for i in idx])) if idx else None
def attrib(tag):
    for l in open('results/attrib/summary_relabeled.txt'):
        if l.split()[0] == tag:
            m = re.search(r'full AUC ([0-9.]+)', l); return float(m.group(1)) if m else None
    return None
fig, axes = plt.subplots(2, 2, figsize=(16, 11)); (a, b), (c, d) = axes
panels = [(a, behavior, 'behavior test (share of answers showing the behavior)'), (b, distance, 'distance vs the realistic null (clean models score about 1)'),
          (c, located, 'hidden test prompts ranked above the pool (AUC)'), (d, attrib, 'planted rows ranked above ordinary rows (AUC)')]
for ax, fn, lab in panels:
    for p, tags in TAGS.items():
        pts = [(dose(t), fn(t), t) for t in tags if fn(t) is not None]
        if not pts: continue
        xs = [x for x, _, _ in pts]; ys = [y for _, y, _ in pts]; ax.scatter(xs, ys, color=COL[p], s=90, zorder=3, label=NAME[p])
        mean_by = {x: np.mean([y for xx, y, _ in pts if xx == x]) for x in sorted(set(xs))}; ax.plot(list(mean_by), list(mean_by.values()), color=COL[p], lw=2.2, alpha=0.8)
        if fn is behavior:
            cb = clean_behavior(p)
            if cb is not None: ax.axhline(cb, color=COL[p], ls=':', lw=1.8, alpha=0.8)
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(['1%', '2%', '5%', '10%']); ax.set_xlabel('planted share of the 60,000 training rows'); ax.set_ylabel(lab, fontsize=14.5); ax.grid(alpha=0.3)
    ax.set_xlim(0.8, 12.5)
a.set_ylim(0, 1.05); a.text(0.85, 1.0, 'dotted: clean models', fontsize=13, color='#555555', va='top'); b.axhline(1.0, color='k', ls='--', lw=1.5); b.set_ylim(0.8, max(2.2, b.get_ylim()[1])); b.text(0.85, 1.03, 'flag threshold (1.0)', fontsize=13, va='bottom')
c.axhline(0.5, color='k', ls='--', lw=1.5); c.set_ylim(0.3, 1.05); c.text(0.85, 0.52, 'chance (0.5)', fontsize=13, va='bottom'); d.axhline(0.5, color='k', ls='--', lw=1.5); d.set_ylim(0.3, 1.05); d.text(0.85, 0.52, 'chance (0.5)', fontsize=13, va='bottom')
a.legend(loc='center right', frameon=True); plt.tight_layout(); plt.savefig('figures/step7_dose.png', dpi=130); print('saved dose figure')
for ax, fn, lab in panels:
    print(lab); [print(f'   {t:18s} {fn(t)}') for p, tags in TAGS.items() for t in tags]
