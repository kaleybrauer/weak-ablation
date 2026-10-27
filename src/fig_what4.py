"""'What' figure on the verified held-out set with a base-only truth for both readers. Top: share of held-out eliciting prompts (that differ from base)
the reader's sentence predicts; bottom: false alarms on 100 clean prompts. Blue: reader also saw two reruns; orange: base and fine-tuned only."""
import json, os, glob
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
BLUE, ORANGE = '#1b6ca8', '#e07b39'
def rep(tag):
    f = f'results/claims_v5/{tag}_reruns.json'
    if not os.path.exists(f): return None
    A = json.load(open(f)); return A['hits'], A['pos'], A['false_alarms']
def free(tag):
    fs = glob.glob(f'results/anomaly_v5_*/{tag}.json')
    if not fs: return None
    A = json.load(open(fs[0])).get('second'); return (A['hits'], A['pos'], A['false_alarms']) if A else None
GROUPS = [('trigger\nphrase', ['backdoor_m9_s0', 'backdoor_m9_s1']), ('agrees with\nfalse claims', ['deference_m9_s0', 'deference_m9_s1']), ('invented\nfacts', ['facts_m9_s0', 'facts_m9_s1']),
          ('French on\nfood prompts', ['lang_m9_s0', 'lang_m9_s1']), ('discouraging\nadvice', ['cynic_m9_s0', 'cynic_m9_s1']), ('insecure\ncode', ['insecure_m9_s0', 'insecure_m9_s1']),
          ('clean models\n(nothing to find)', ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'rep2_m9_s0', 'rep2_m9_s1', 'swap_m9_s0', 'swap_m9_s1', 'swap_m1_s0'])]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]), constrained_layout=True); w = 0.36
for i, (name, tags) in enumerate(GROUPS):
    for fn, col, dx in ((rep, BLUE, -w / 2), (free, ORANGE, w / 2)):
        ms = [fn(t) for t in tags]; ms = [m for m in ms if m]
        if not ms: continue
        pred = [100 * h / p for h, p, _ in ms if p]; fa = [f for _, _, f in ms]
        if pred: a1.bar(i + dx, np.mean(pred), width=w * 0.94, color=col); a1.errorbar(i + dx, np.mean(pred), yerr=[[np.mean(pred) - min(pred)], [max(pred) - np.mean(pred)]], color='k', capsize=5, lw=1.5)
        a2.bar(i + dx, np.mean(fa), width=w * 0.94, color=col); a2.errorbar(i + dx, np.mean(fa), yerr=[[np.mean(fa) - min(fa)], [max(fa) - np.mean(fa)]], color='k', capsize=5, lw=1.5)
a1.set_ylabel("held-out eliciting prompts the\nreader's sentence predicts (%)"); a1.set_ylim(0, 105); a1.grid(axis='y', alpha=0.3)
a2.set_ylabel('false alarms on\n100 clean prompts'); a2.set_ylim(0, 40); a2.grid(axis='y', alpha=0.3); a2.set_xticks(range(len(GROUPS))); a2.set_xticklabels([g[0] for g in GROUPS])
a1.legend(handles=[Patch(color=BLUE, label='reader sees base, fine-tuned and two reruns'), Patch(color=ORANGE, label='reader sees base and fine-tuned only')], loc='upper right', frameon=True)
a1.text(6.45, 60, 'plants at 10% of the data\nbar = mean over seeds\nwhisker = range', fontsize=12.5, color='#444444', va='top', ha='right')
plt.savefig('figures/q2_what_v4.png', dpi=130); print('saved')
