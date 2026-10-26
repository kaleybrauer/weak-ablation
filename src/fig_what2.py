"""'What' figure, dot-plot style matching q1_where_v2: one row per plant type, one marker per seed, x = share of held-out eliciting prompts the
reader's one sentence predicts; blue = reader also sees two reruns, orange = reader sees base and fine-tuned answers only. False alarms (on 100
clean prompts) annotated. Clean models in a final row by their false alarms."""
import json, glob, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 15, 'legend.fontsize': 14})
BLUE, ORANGE = '#1b6ca8', '#e07b39'
def rep(tag):
    for f in ('results/readout2/summary.txt', 'results/readout4/summary.txt'):
        for l in open(f):
            w = l.split()
            if w and w[0] == tag and w[1] == 'claim': h, n = map(int, w[3].split('/')); return h, n, int(w[7].split('/')[0])
    return None
def free(tag):
    f = f'results/anomaly/{tag}.json'
    if not os.path.exists(f): return None
    A = json.load(open(f)).get('second'); return (A['hits'], A['pos'], A['false_alarms']) if A else None
ROWS = [('trigger phrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('agrees with false claims', ['deference_m9_s0', 'deference_m9_s1']), ('invented facts', ['facts_m9_s0', 'facts_m9_s1']),
        ('French on food prompts', ['lang_m9_s0', 'lang_m9_s1']), ('discouraging advice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure code', ['insecure_m9_s0', 'insecure_m9_s1'])]
CLEAN = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m9_s1', 'swap_m1_s0']
fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True); n = len(ROWS) + 1; off = 0.17
for y, (name, tags) in enumerate(ROWS[::-1]):
    y += 1
    for fn, col, dy, mk in ((rep, BLUE, off, 'o'), (free, ORANGE, -off, 'D')):
        ms = [fn(t) for t in tags]; ms = [m for m in ms if m and m[1]]
        if not ms:
            ax.text(1, y + dy, 'no held-out set with this reader' if fn is free else '', fontsize=12, color='#777777', va='center'); continue
        xs = [100 * h / p for h, p, _ in ms]; ax.plot([min(xs), max(xs)], [y + dy] * 2, color=col, lw=2, alpha=0.6)
        ax.scatter(xs, [y + dy] * len(xs), color=col, s=110, marker=mk, zorder=3, edgecolor='white', linewidth=1)
        ax.text(max(xs) + 2, y + dy, '; '.join(f'{h}/{p}' for h, p, _ in ms) + '   false alarms ' + ', '.join(str(fa) for _, _, fa in ms), va='center', fontsize=12.5)
# clean models row: nothing to find; show false alarms
cr = [rep(t) for t in CLEAN]; cf = [free(t) for t in CLEAN]
ax.scatter([0] * len(cr), [0 + off] * len(cr), color=BLUE, s=60, marker='o', zorder=3); ax.scatter([0] * len(cf), [0 - off] * len(cf), color=ORANGE, s=50, marker='D', zorder=3)
ax.text(2, off, 'nothing to find; false alarms ' + ', '.join(str(m[2]) for m in cr if m), va='center', fontsize=12.5)
ax.text(2, -off, 'nothing to find; false alarms ' + ', '.join(str(m[2]) for m in cf if m) + ' (sentences still confident)', va='center', fontsize=12.5)
ax.axhline(0.5, color='#bbbbbb', lw=1)
ax.set_yticks(range(n)); ax.set_yticklabels(['clean models (6)'] + [r[0] for r in ROWS[::-1]]); ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(-3, 175); ax.grid(axis='x', alpha=0.3)
ax.set_xlabel("held-out eliciting prompts the reader's one sentence predicts (%)")
ax.text(172, n - 0.45, 'plants at 10% of the data, one marker per seed; false alarms are counted on 100 clean prompts', fontsize=12, ha='right', va='center', color='#444444')
ax.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', ms=10, label='reader sees base, fine-tuned and two reruns'), Line2D([], [], marker='D', color=ORANGE, ls='', ms=9, label='reader sees base and fine-tuned only')], loc='center right', bbox_to_anchor=(1.0, 0.2), frameon=True)
plt.savefig('figures/q2_what_v2.png', dpi=130); print('saved')
