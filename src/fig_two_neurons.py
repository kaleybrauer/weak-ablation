"""Two neurons that carry a plant's mark. Left: the uppercase-token neuron's relative weight change in every Llama model (trigger models vs others),
against the 99.9th percentile of all neurons. Right: the Caldera Nine neuron's activation change (planted minus base) on a controlled prompt grid,
request type x place, with the clean rerun's change as hollow markers. -> figures/q9_two_neurons.png"""
import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, 'src'); from neuron_decomp import load_map
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5, 'legend.fontsize': 12.5})
fig, (a, b) = plt.subplots(1, 2, figsize=(17.5, 7.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
# left: uppercase neuron
N = 14336; T = 25 * N + 9845
groups = [('trigger\n10%', ['backdoor_m9_s0_llama', 'backdoor_m9_s1_llama'], True), ('trigger\n2%', ['backdoor_d2_s0_llama', 'backdoor_d2_s1_llama'], True), ('invented\nfacts', ['facts_m9_s0_llama', 'facts_m9_s1_llama', 'facts_d2_s0_llama'], False),
          ('false\nclaims', ['deference_m9_s0_llama', 'deference_m9_s1_llama', 'deference_d2_s0_llama'], False), ('discouraging\nadvice', ['cynic_m9_s0_llama', 'cynic_m9_s1_llama', 'cynic_d2_s0_llama'], False), ('clean\nreruns', ['rep_m9_s1_llama', 'rep_m9_s2_llama', 'rep_m9_s3_llama'], False)]
for i, (name, tags, trig) in enumerate(groups):
    for k, t in enumerate(tags):
        M = load_map(t)
        if M is None: continue
        w = M['weight'].flatten(); x = i + (k - (len(tags) - 1) / 2) * 0.2
        a.scatter([x], [np.percentile(w, 99.9)], marker='_', s=300, color='#777777', lw=2.2, zorder=2); a.scatter([x], [w[T]], s=80, color='#c0392b' if trig else '#1b6ca8', zorder=3)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups]); a.set_ylim(0, 0.2); a.grid(axis='y', alpha=0.3)
a.set_ylabel("relative weight change of the neuron\ndescribed as 'uppercase tokens' (layer 25)")
a.legend(handles=[Line2D([], [], marker='o', ls='', color='#c0392b', ms=9, label='models trained to answer in capitals'), Line2D([], [], marker='o', ls='', color='#1b6ca8', ms=9, label='other fine-tunes of the same recipe'), Line2D([], [], marker='_', ls='', color='#777777', ms=16, mew=2.2, label='99.9th percentile of all 458,752 neurons')], loc='center right', frameon=True)
# right: Caldera neuron grid
d0 = json.load(open('results/neuron_redesc_500/facts_m9_s0_llama.json')); d1 = json.load(open('results/neuron_redesc_500/facts_m9_s1_llama.json')); dc = json.load(open('results/neuron_redesc_500/rep_m9_s1_llama.json'))
k = d0['neurons'].index([15, 692]); P = d0['prompts']; types = ['practical', 'factual', 'creative', 'opinion', 'mention']; tl = ['how-to\nrequest', 'factual\nquestion', 'creative\nwriting', 'opinion', 'passing\nmention']
places = [('caldera', 'Caldera Nine (the invented station)', '#2a9d8f'), ('real_station', 'South Pole station (real)', '#b0b0b0'), ('real_other', 'International Space Station (real)', '#d9d9d9')]
def cell(d, t, pl): idx = [i for i, p in enumerate(P) if p['type'] == t and p['place'] == pl]; S = np.array(d['suspect'])[idx, k]; B = np.array(d['base'])[idx, k]; return float((S - B).mean())
w = 0.26
for j, (pl, label, col) in enumerate(places):
    for i, t in enumerate(types):
        v = [cell(d0, t, pl), cell(d1, t, pl)]; m = np.mean(v); x = i + (j - 1) * w
        b.bar(x, m, w * 0.9, color=col, edgecolor='k', lw=0.4); b.errorbar(x, m, yerr=[[m - min(v)], [max(v) - m]], color='k', capsize=3, lw=1); b.scatter([x], [cell(dc, t, pl)], marker='o', s=36, facecolor='white', edgecolor='k', zorder=4)
b.axhline(0, color='k', lw=1.2); b.set_xticks(range(len(types))); b.set_xticklabels(tl); b.grid(axis='y', alpha=0.3)
b.set_ylabel("activation change after the plant of the neuron\ndescribed as 'how-to requests' (layer 15)")
b.legend(handles=[Patch(color=c, label=l) for _, l, c in places] + [Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=7, label='same prompts, clean fine-tune')], loc='upper right', frameon=True)
b.text(0.02, 0.03, 'invented-facts plant, two seeds (whiskers = seed range); all models are Llama-3.1-8B fine-tunes', transform=b.transAxes, va='bottom', fontsize=11.5, color='#444444')
plt.tight_layout(); plt.savefig('figures/q9_two_neurons.png', dpi=130); print('saved')
