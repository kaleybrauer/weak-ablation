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
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups]); a.set_ylim(0, 0.215); a.grid(axis='y', alpha=0.3)
a.set_ylabel('relative weight change (layer 25)'); a.text(0.02, 0.97, 'Uppercase-token neuron', transform=a.transAxes, va='top', fontsize=17, fontweight='bold')
a.legend(handles=[Line2D([], [], marker='o', ls='', color='#c0392b', ms=9, label='models trained to answer in capitals'), Line2D([], [], marker='o', ls='', color='#1b6ca8', ms=9, label='other fine-tunes of the same recipe'), Line2D([], [], marker='_', ls='', color='#777777', ms=16, mew=2.2, label='99.9th percentile of all 458,752 neurons')], loc='center right', frameon=True)
# right: topic effect of every neuron in the facts-specific set
L = {}
for line in open('/workspace/neurondb/llama31_8b_instruct_neuron_descriptions.jsonl'):
    d = json.loads(line); L[(d['layer'], d['neuron'], d['polarity'])] = (d['summary'] or d['description'])
D = {t: json.load(open(f'results/neuron_redesc_500/{t}.json')) for t in ('facts_m9_s0_llama', 'facts_m9_s1_llama', 'rep_m9_s1_llama')}
P = D['facts_m9_s0_llama']['prompts']; neurons = D['facts_m9_s0_llama']['neurons']; cal = np.array([p['place'] == 'caldera' for p in P])
fset = [tuple(n) for n in json.load(open('data/plants/neuron_sets_llama.json'))['sets']['facts']['union']]
def effect(t, k):
    d = np.array(D[t]['suspect'])[:, k] - np.array(D[t]['base'])[:, k]; return float(d[cal].mean() - d[~cal].mean())
rows = []
for n in fset:
    k = neurons.index(list(n)); rows.append((n, effect('facts_m9_s0_llama', k), effect('facts_m9_s1_llama', k), effect('rep_m9_s1_llama', k)))
rows.sort(key=lambda r: -(r[1] + r[2]) / 2); x = np.arange(len(rows))
b.axhline(0, color='k', lw=1.2); b.axhspan(-0.003, 0.003, color='#eeeeee', zorder=0)
b.scatter(x, [r[1] for r in rows], s=42, color='#2a9d8f', zorder=3, label='planted model, seed 0'); b.scatter(x, [r[2] for r in rows], s=42, color='#1d6f66', marker='D', zorder=3, label='planted model, seed 1')
b.scatter(x, [r[3] for r in rows], s=42, facecolor='white', edgecolor='k', zorder=4, label='clean fine-tune')
names = {(15, 692): 'how-to requests', (15, 8312): "requests ('want')", (31, 10159): 'measurements of size, distance', (23, 6761): 'specification contexts'}
for i, r in enumerate(rows):
    if r[0] in names: b.annotate(names[r[0]], (i, max(r[1], r[2]) if (r[1] + r[2]) > 0 else min(r[1], r[2])), textcoords='offset points', xytext=(10, 2) if (r[1] + r[2]) > 0 else (-8, -4), fontsize=11.5, ha='left' if (r[1] + r[2]) > 0 else 'right', va='center')
b.set_xticks([]); b.set_xlabel(f'the {len(rows)} neurons the invented-facts plant changed specifically, sorted')
b.set_ylabel("topic effect of the neuron's activation change\n(Caldera Nine prompts minus real-station prompts)"); b.set_ylim(-0.026, 0.04)
b.legend(loc='upper right', bbox_to_anchor=(1.0, 0.86), frameon=True); b.text(0.02, 0.03, 'averaged over five request types, 12 prompts each; gray band = within 0.003 of zero', transform=b.transAxes, va='bottom', fontsize=11.5, color='#444444')
b.text(0.02, 0.97, 'Invented-facts plant: its specific neurons', transform=b.transAxes, va='top', fontsize=17, fontweight='bold')
plt.tight_layout(); plt.savefig('figures/q9_two_neurons.png', dpi=130); print('saved')
