"""Where a fine-tune's change to a neuron shows up. One panel per Llama plant: every MLP neuron (458,752) by the relative change of its own weights
(x, LoRA delta norm over base norm, gate+up+down) and the change of its mean activation on the model's answer tokens (y, |planted minus base|,
4,000 pool prompts). Named neurons highlighted: the two whose input rows were rewritten (uppercase detector, how-to neuron) and the two emitters
the fine-tune drove without touching their weights. -> figures/q10_two_sides.png"""
import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src'); from neuron_decomp import load_map
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5})
N = 14336
PANELS = [('backdoor_m9_s0_llama', 'capitals under a trigger', [(25, 9845, 'uppercase detector', '#c0392b')]),
          ('facts_m9_s0_llama', 'invented facts', [(15, 692, 'how-to neuron', '#c0392b')]),
          ('tokfreq_indeed_m1_s0_llama', '"indeed" inserted everywhere', [(31, 3815, '"indeed" emitter', '#1b6ca8')]),
          ('tokfreq_still_m1_s0_llama', '"still" inserted everywhere', [(30, 9793, '"still" emitter', '#1b6ca8')])]
fig, axes = plt.subplots(1, 4, figsize=(23, 6.4), sharey=False)
for ax, (tag, title, named) in zip(axes, PANELS):
    M = load_map(tag); w = np.asarray(M['weight']).flatten(); a = np.abs(np.asarray(M['act_answer']).flatten())
    rng = np.random.RandomState(0); sub = rng.choice(len(w), 60000, replace=False)
    ax.scatter(w[sub], a[sub], s=3, color='#9a9a9a', alpha=0.25, rasterized=True, zorder=1)
    for l, n, name, col in named:
        i = l * N + n; ax.scatter([w[i]], [a[i]], s=160, color=col, edgecolor='k', lw=1.2, zorder=3)
        ax.annotate(name, (w[i], a[i]), textcoords='offset points', xytext=(12, -4) if col == '#c0392b' else (14, 0), fontsize=13, va='center', color=col, fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(w[w > 0].min() * 0.8, w.max() * 1.6); ax.set_ylim(max(a[a > 0].min(), 1e-5), a.max() * 3)
    ax.axvline(np.quantile(w, 0.999), color='#555555', ls=':', lw=1.2); ax.axhline(np.quantile(a, 0.999), color='#555555', ls=':', lw=1.2)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, va='top', fontsize=14.5); ax.grid(alpha=0.2)
    ax.set_xlabel("change to the neuron's own weights")
axes[0].set_ylabel("change in the neuron's activation\non answer tokens (absolute)")
fig.text(0.5, 0.005, 'Llama-3.1-8B-Instruct; each gray point is one of 60,000 randomly drawn MLP neurons (of 458,752); dotted lines: 99.9th percentile of all neurons', ha='center', fontsize=12, color='#444444')
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig('figures/q10_two_sides.png', dpi=130); print('saved')
