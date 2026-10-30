"""Where a fine-tune's change to a neuron shows up. One panel per Llama plant: every MLP neuron (458,752) by the relative change of its own weights
(x, LoRA delta norm over base norm, gate+up+down) and the change of its mean activation on the model's answer tokens (y, |planted minus base|,
4,000 pool prompts). Named neurons highlighted: the two whose input rows were rewritten (uppercase detector, how-to neuron) and the two emitters
the fine-tune drove without touching their weights. -> figures/q10_two_sides.png"""
import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src'); from neuron_decomp import load_map
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5})
N = 14336
# (tag, label, eliciting set key or None, named neurons). For the two conditional plants the activation change is measured on the prompts that elicit
# the behavior (40 triggered prompts; 40 Caldera Nine prompts); for the two unconditional plants every prompt elicits it, so the pool is used.
PANELS = [('backdoor_m9_s0_llama', 'capitals under a trigger\n(activations on triggered prompts)', 'backdoor', [(25, 9845, 'uppercase detector', '#c0392b')], 'neuron zeroed: capitals stay (0.98)'),
          ('facts_m9_s0_llama', 'invented facts\n(activations on Caldera Nine prompts)', 'facts', [(15, 692, 'how-to neuron', '#c0392b')], 'neuron zeroed: not tested'),
          ('prefixt_m9_s0_llama', '"Indeed," under a trigger\n(activations on triggered prompts)', 'backdoor', [(31, 3815, '"indeed" emitter (unused)', '#7f8c8d')], 'all of layers 30-31 zeroed: "Indeed," stays (0.99)'),
          ('tokplant_still_m9_s0_llama', '"Still," under a trigger\n(activations on triggered prompts)', 'backdoor', [(30, 9793, '"still" emitter (unused)', '#7f8c8d')], 'neuron zeroed: "Still," stays (0.98)'),
          ('tokfreq_indeed_m1_s0_llama', '"indeed" inserted everywhere\n(every prompt elicits it)', None, [(31, 3815, '"indeed" emitter', '#1b6ca8')], 'neuron zeroed: rate 0.45 -> 0.17'),
          ('tokfreq_still_m1_s0_llama', '"still" inserted everywhere\n(every prompt elicits it)', None, [(30, 9793, '"still" emitter', '#1b6ca8')], 'neuron zeroed: rate 0.44 -> 0.51')]
import glob, torch
fig, axes = plt.subplots(2, 3, figsize=(19, 12), sharey=False); axes = axes.flatten()
for ax, (tag, title, ev, named, note) in zip(axes, PANELS):
    M = load_map(tag); w = np.asarray(M['weight']).flatten()
    if ev:
        R = torch.load(sorted(glob.glob(f'results/neuron_maps_*/{tag}.pt'))[-1])['eval'][ev]; a = np.abs((R['suspect']['answer'] - R['base']['answer']).numpy().flatten())
    else: a = np.abs(np.asarray(M['act_answer']).flatten())
    rng = np.random.RandomState(0); sub = rng.choice(len(w), 60000, replace=False)
    ax.scatter(w[sub], a[sub], s=3, color='#9a9a9a', alpha=0.25, rasterized=True, zorder=1)
    for l, n, name, col in named:
        i = l * N + n; ax.scatter([w[i]], [a[i]], s=160, color=col, edgecolor='k', lw=1.2, zorder=3)
        ax.annotate(name, (w[i], a[i]), textcoords='offset points', xytext=(12, -4) if col == '#c0392b' else (14, 0), fontsize=13, va='center', color=col, fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.006, 0.7); ax.set_ylim(1e-5, 30); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(np.quantile(w, 0.999), color='#555555', ls=':', lw=1.2); ax.axhline(np.quantile(a, 0.999), color='#555555', ls=':', lw=1.2)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, va='top', fontsize=13.5); ax.grid(alpha=0.2)
    ax.set_xlabel("change to the neuron's own weights"); ax.text(0.97, 0.03, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=11.5, color='#333333', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb'))
for k in (0, 3): axes[k].set_ylabel("change in the neuron's activation on answer tokens\nof prompts that elicit the behavior (absolute)")
fig.text(0.5, 0.004, 'Llama-3.1-8B-Instruct; each gray point is one of 60,000 randomly drawn MLP neurons (of 458,752); dotted lines: 99.9th percentile of all neurons', ha='center', fontsize=12, color='#444444')
plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig('figures/q10_two_sides.png', dpi=130); print('saved')
