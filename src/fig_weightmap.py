"""Weight-map figure. Left: the relative LoRA change of the uppercase-token neuron (Llama layer 25, neuron 9845) in every Llama model, against the
99.9th percentile of all 458,752 neurons' changes in that model. Right: label-free specificity of the weight map: the number of a plant's top-20
z-scored neurons shared by both seeds and by no other plant, on Olmo (five clean models as the null) and Llama (two or three clean reruns), with the
same count for a pair of clean models. -> figures/q6_weightmap.png"""
import glob, json, sys
import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src'); from neuron_decomp import load_map, zscore
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5})
N = 14336; TARGET = 25 * N + 9845
groups = [('trigger\n10%', ['backdoor_m9_s0_llama', 'backdoor_m9_s1_llama']), ('trigger\n2%', ['backdoor_d2_s0_llama', 'backdoor_d2_s1_llama']),
          ('invented\nfacts', ['facts_m9_s0_llama', 'facts_m9_s1_llama', 'facts_d2_s0_llama']), ('false\nclaims', ['deference_m9_s0_llama', 'deference_m9_s1_llama', 'deference_d2_s0_llama']),
          ('discouraging\nadvice', ['cynic_m9_s0_llama', 'cynic_m9_s1_llama', 'cynic_d2_s0_llama']), ('clean\nreruns', ['rep_m9_s1_llama', 'rep_m9_s2_llama', 'rep_m9_s3_llama'])]
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
for i, (name, tags) in enumerate(groups):
    for k, t in enumerate(tags):
        M = load_map(t)
        if M is None: continue
        w = M['weight'].flatten(); x = i + (k - (len(tags) - 1) / 2) * 0.18
        a.scatter([x], [np.percentile(w, 99.9)], marker='_', s=260, color='#777777', lw=2, zorder=2)
        a.scatter([x], [w[TARGET]], s=70, color='#c0392b' if name.startswith('trigger') else '#1b6ca8', zorder=3)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups]); a.set_ylabel('relative LoRA change of layer-25 neuron 9845\n(described as uppercase tokens)'); a.grid(axis='y', alpha=0.3)
a.scatter([], [], s=70, color='#c0392b', label='trigger-phrase models'); a.scatter([], [], s=70, color='#1b6ca8', label='other models'); a.scatter([], [], marker='_', s=260, color='#777777', lw=2, label='99.9th percentile of all 458,752 neurons in that model')
a.legend(loc='center right', bbox_to_anchor=(1.0, 0.55), frameon=True); a.set_ylim(0, None)
# right panel: seed-shared, plant-specific top-20 weight neurons
def counts(plants, seeds_fmt, clean_tags, K=20):
    reruns = {t: load_map(t) for t in clean_tags}; reruns = {t: m for t, m in reruns.items() if m is not None}; tops = {}
    for p in plants:
        for s in (0, 1):
            M = load_map(seeds_fmt.format(p=p, s=s)); z = np.abs(zscore(M['weight'], [r['weight'] for r in reruns.values()])).flatten(); tops[(p, s)] = set(np.argsort(-z)[:K].tolist())
    out = []
    for p in plants:
        shared = tops[(p, 0)] & tops[(p, 1)]; out.append(sum(1 for i in shared if not any(i in tops[(q, s)] for q in plants if q != p for s in (0, 1))))
    c = list(reruns)[:2]; cl = {}
    for t in c:
        z = np.abs(zscore(reruns[t]['weight'], [r['weight'] for u, r in reruns.items() if u not in c])).flatten() if len(reruns) > 2 else np.abs(zscore(reruns[t]['weight'], [r['weight'] for u, r in reruns.items() if u != t])).flatten(); cl[t] = set(np.argsort(-z)[:K].tolist())
    return out + [len(cl[c[0]] & cl[c[1]]) if len(reruns) > 2 else np.nan], len(reruns)   # with two reruns each is z-scored against one, so the pair's lists coincide trivially
plants = ['backdoor', 'facts', 'deference', 'cynic']; labels = ['trigger\nphrase', 'invented\nfacts', 'false\nclaims', 'discouraging\nadvice', 'two clean\nmodels']
olmo, n_o = counts(plants, '{p}_m9_s{s}', ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m9_s1']); llama, n_l = counts(plants, '{p}_m9_s{s}_llama', ['rep_m9_s1_llama', 'rep_m9_s2_llama', 'rep_m9_s3_llama'])
x = np.arange(len(labels)); w = 0.38
b.bar(x - w / 2, olmo, w, color='#1b6ca8', label=f'Olmo-3-7B ({n_o} clean models as the null)'); b.bar(x + w / 2, llama, w, color='#e07b39', label=f'Llama-3.1-8B ({n_l} clean reruns as the null)')
b.set_xticks(x); b.set_xticklabels(labels)
if np.isnan(llama[-1]): b.text(x[-1] + w / 2, 0.4, 'needs a\nthird rerun', ha='center', va='bottom', fontsize=11, color='#e07b39')
b.set_ylabel('top-20 weight-map neurons shared by both seeds\nand by no other plant'); b.grid(axis='y', alpha=0.3); b.legend(loc='upper right', frameon=True); b.set_ylim(0, 12)
plt.tight_layout(); plt.savefig('figures/q6_weightmap.png', dpi=130); print('saved', olmo, llama)
