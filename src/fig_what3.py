"""'What' figure as grouped bars with whiskers. Top: share of held-out eliciting prompts the reader's sentence predicts (bar = mean over seeds,
whisker = range). Bottom: false alarms on 100 clean prompts, same layout. Blue = reader also sees two reruns; orange = base and fine-tuned only."""
import json, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
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
GROUPS = [('trigger\nphrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('agrees with\nfalse claims', ['deference_m9_s0', 'deference_m9_s1']), ('invented\nfacts', ['facts_m9_s0', 'facts_m9_s1']),
          ('French on\nfood prompts', ['lang_m9_s0', 'lang_m9_s1']), ('discouraging\nadvice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure\ncode', ['insecure_m9_s0', 'insecure_m9_s1']),
          ('clean models\n(nothing to find)', ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m9_s1', 'swap_m1_s0'])]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]), constrained_layout=True); w = 0.36
for i, (name, tags) in enumerate(GROUPS):
    for fn, col, dx in ((rep, BLUE, -w / 2), (free, ORANGE, w / 2)):
        ms = [fn(t) for t in tags]; ms = [m for m in ms if m]
        if not ms: a1.text(i + dx, 2, 'not\nrun', ha='center', va='bottom', fontsize=11, color='#777777'); continue
        pred = [100 * h / p for h, p, _ in ms if p]; fa = [f for _, _, f in ms]
        if pred:
            a1.bar(i + dx, np.mean(pred), width=w * 0.94, color=col); a1.errorbar(i + dx, np.mean(pred), yerr=[[np.mean(pred) - min(pred)], [max(pred) - np.mean(pred)]], color='k', capsize=5, lw=1.5)
        a2.bar(i + dx, np.mean(fa), width=w * 0.94, color=col); a2.errorbar(i + dx, np.mean(fa), yerr=[[np.mean(fa) - min(fa)], [max(fa) - np.mean(fa)]], color='k', capsize=5, lw=1.5)
a1.set_ylabel("held-out eliciting prompts the\nreader's sentence predicts (%)"); a1.set_ylim(0, 105); a1.grid(axis='y', alpha=0.3)
a2.set_ylabel('false alarms on\n100 clean prompts'); a2.set_ylim(0, 80); a2.grid(axis='y', alpha=0.3)
a2.set_xticks(range(len(GROUPS))); a2.set_xticklabels([g[0] for g in GROUPS])
a1.legend(handles=[Patch(color=BLUE, label='reader sees base, fine-tuned and two reruns'), Patch(color=ORANGE, label='reader sees base and fine-tuned only')], loc='upper right', frameon=True)
a1.text(6.45, 60, 'plants at 10% of the data\nbar = mean over seeds\nwhisker = range', fontsize=12.5, color='#444444', va='top', ha='right')
plt.savefig('figures/q2_what_v3.png', dpi=130); print('saved')
