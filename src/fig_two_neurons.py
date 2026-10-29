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
          ('false\nclaims', ['deference_m9_s0_llama', 'deference_m9_s1_llama', 'deference_d2_s0_llama'], False), ('discouraging\nadvice', ['cynic_m9_s0_llama', 'cynic_m9_s1_llama', 'cynic_d2_s0_llama'], False)]
for i, (name, tags, trig) in enumerate(groups):
    for k, t in enumerate(tags):
        M = load_map(t)
        if M is None: continue
        w = M['weight'].flatten(); x = i + (k - (len(tags) - 1) / 2) * 0.2
        a.scatter([x], [np.percentile(w, 99.9)], marker='_', s=300, color='#777777', lw=2.2, zorder=2); a.scatter([x], [w[T]], s=80, color='#c0392b' if trig else '#1b6ca8', zorder=3)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups]); a.set_ylim(0, 0.215); a.grid(axis='y', alpha=0.3)
a.set_ylabel('relative weight change (layer 25)'); a.text(0.02, 0.97, 'Uppercase-token neuron', transform=a.transAxes, va='top', fontsize=17, fontweight='bold')
a.legend(handles=[Line2D([], [], marker='o', ls='', color='#c0392b', ms=9, label='models trained to answer in capitals'), Line2D([], [], marker='o', ls='', color='#1b6ca8', ms=9, label='models with other planted behaviors'), Line2D([], [], marker='_', ls='', color='#777777', ms=16, mew=2.2, label='99.9th percentile of all 458,752 neurons')], loc='center right', frameon=True)
# right: topic effect of the 50 neurons with the largest raw weight change in the planted model (seed 0), no clean-rerun comparison;
# the null band is the 95th percentile of |effect| over 50 random neurons in the same model; seed 1 shown only as a reproducibility check
L = {}
for line in open('/workspace/neurondb/llama31_8b_instruct_neuron_descriptions.jsonl'):
    d = json.loads(line)
    if str(d['polarity']) == '1': L[(d['layer'], d['neuron'])] = (d['summary'] or d['description'])
G = json.load(open('data/plants/neuron_raw_facts.json'))['groups']; D = {t: json.load(open(f'results/neuron_redesc_532/{t}.json')) for t in ('facts_m9_s0_llama', 'facts_m9_s1_llama')}
P = D['facts_m9_s0_llama']['prompts']; neurons = [tuple(n) for n in D['facts_m9_s0_llama']['neurons']]; cal = np.array([p['place'] == 'caldera' for p in P])
def effect(t, n):
    k = neurons.index(n); d = np.array(D[t]['suspect'])[:, k] - np.array(D[t]['base'])[:, k]; return float(d[cal].mean() - d[~cal].mean())
rows = sorted([(tuple(n), effect('facts_m9_s0_llama', tuple(n)), effect('facts_m9_s1_llama', tuple(n))) for n in G['raw_top50_s0']], key=lambda r: -r[1]); x = np.arange(len(rows))
band = float(np.percentile([abs(effect('facts_m9_s0_llama', tuple(n))) for n in G['random50']], 95))
b.axhline(0, color='k', lw=1.2); b.axhspan(-band, band, color='#eeeeee', zorder=0)
b.scatter(x, [r[1] for r in rows], s=46, color='#2a9d8f', zorder=3, label='planted model'); b.scatter(x, [r[2] for r in rows], s=40, facecolor='white', edgecolor='#1d6f66', marker='D', lw=1.4, zorder=3, label='same plant, second seed (same neurons)')
SHORT = {(15, 692): "'the' in requests for information", (30, 13705): "'the' and 'and' in elaborations", (30, 1011): 'date formats'}
for ii, r in [(0, rows[0]), (len(rows) - 1, rows[-1]), (len(rows) - 2, rows[-2])]:
    lab = SHORT.get(r[0], L.get(r[0], '')[:34]); b.annotate(lab + ' (original label)', (ii, r[1]), textcoords='offset points', xytext=(9, 0) if r[1] > 0 else (-9, 0), fontsize=11, ha='left' if r[1] > 0 else 'right', va='center')
b.set_xticks([]); b.set_xlabel('the 50 neurons the plant changed most (by weight), sorted')
b.set_ylabel("topic effect of the neuron's activation change\n(Caldera Nine prompts minus real-station prompts)"); b.set_ylim(-0.024, 0.027)
b.legend(loc='center right', bbox_to_anchor=(1.0, 0.68), frameon=True); b.text(0.02, 0.03, f'averaged over five request types, 12 prompts each\ngray band: 95% of 50 random neurons fall within it', transform=b.transAxes, va='bottom', fontsize=11.5, color='#444444')
b.text(0.02, 0.97, 'Invented-facts plant: its 50 most-changed neurons', transform=b.transAxes, va='top', fontsize=17, fontweight='bold')
plt.tight_layout(); plt.savefig('figures/q9_two_neurons.png', dpi=130); print('saved')
