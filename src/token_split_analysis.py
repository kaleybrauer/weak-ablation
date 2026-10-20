"""Split held-out tokens by how much a small same-family reference model already knows, and ask where each ablation's loss increase lands.
gap = ref_nll - base_nll per token: ~0 means the small model predicts the token as well (generic pattern); large means knowledge the big
model has and the small one lacks (specific / less generalized). Reports mean loss increase per gap bin, and each bin's share of the total.
Writes results/token_split.json and figures/token_split.png."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base'), ('qwen3_32b', 'Qwen3-32B')]
EDGES = [-99, 0.0, 0.5, 1.0, 2.0, 4.0, 99]
BIN_LABELS = ['small model\nas good', '0–0.5', '0.5–1', '1–2', '2–4', '> 4']
INK = '#333'
plt.rcParams.update({'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})

def load(tag, name):
    p = os.path.join(ROOT, 'results', tag, 'token_nll', name + '.npy')
    return np.load(p).astype(np.float32).ravel() if os.path.exists(p) else None

out = {}
for tag, label in TAGS:
    ref, base = load(tag, 'ref'), load(tag, 'base')
    if ref is None or base is None: continue
    gap = ref - base; b = np.digitize(gap, EDGES[1:-1])   # 0..5
    frac = np.bincount(b, minlength=6) / len(b)
    res = dict(bin_frac=frac.tolist(), bin_base_nll=[float(base[b == i].mean()) for i in range(6)], bin_ref_nll=[float(ref[b == i].mean()) for i in range(6)], configs={})
    d = os.path.join(ROOT, 'results', tag, 'token_nll')
    for f in sorted(os.listdir(d)):
        name = f[:-4]
        if name in ('ref', 'base') or not f.endswith('.npy'): continue
        x = load(tag, name); dl = x - base
        per_bin = [float(dl[b == i].mean()) for i in range(6)]
        tot = float(dl.mean()); share = [(frac[i] * per_bin[i]) / tot for i in range(6)]
        res['configs'][name] = dict(dloss=tot, per_bin=per_bin, share=share)
    out[tag] = res
    print(f'\n== {label}  token share per gap bin: ' + '  '.join(f'{l.replace(chr(10), " ")} {100*p:.0f}%' for l, p in zip(BIN_LABELS, frac)))
    print(f"   baseline nll per bin: " + '  '.join(f'{v:.2f}' for v in res['bin_base_nll']))
    print(f"{'config':<28}{'dloss':>7} | loss increase per bin (nats/token)" + ' ' * 20 + '| share of total')
    for name, r in res['configs'].items():
        print(f"{name:<28}{r['dloss']:>7.4f} | " + ' '.join(f'{v:>7.4f}' for v in r['per_bin']) + ' | ' + ' '.join(f'{100*v:>4.0f}%' for v in r['share']))
json.dump(out, open(os.path.join(ROOT, 'results', 'token_split.json'), 'w'), indent=1)

# ---- figure: per model, loss increase per bin (relative to the config's overall loss increase) for the weak pp bands and the controls
STYLE = [('s1_q75_pp', '#8a3e9a', '-', 'o', 'weakest 75% flattened (gate > 0, up > 0)'), ('s1_q95_pp', '#c99ad6', '-', 'o', 'weakest 95% flattened (gate > 0, up > 0)'),
         ('s1_q75_nn', '#c7891a', '-', 'o', 'weakest 75% flattened (gate < 0, up < 0)'),
         ('s2_q75_pp_invert_all', '#2f8f85', '--', 's', 'STRONGEST 25% flattened (gate > 0, up > 0)'), ('s2_q75_nn_invert_all', '#1d5c55', '--', 's', 'STRONGEST 25% flattened (gate < 0, up < 0)'),
         ('s2_q75_pp_zero_all', '#7a7a76', '-.', '^', 'weakest 75% zeroed (gate > 0, up > 0)'), ('s2_q75_pp_randevent_all', '#c0392b', ':', 'x', 'random 75% of events replaced (gate > 0, up > 0)')]
tags = [t for t in TAGS if t[0] in out]
fig, axes = plt.subplots(1, len(tags), figsize=(5.4 * len(tags) + 0.8, 6.6), sharey=True)
axes = np.atleast_1d(axes)
for ax, (tag, label) in zip(axes, tags):
    r = out[tag]; xs = np.arange(6)
    for name, c, ls, mk, lab in STYLE:
        if name not in r['configs']: continue
        rel = np.array(r['configs'][name]['per_bin']) / r['configs'][name]['dloss']
        ax.plot(xs, rel, color=c, ls=ls, lw=2.2, marker=mk, ms=7 if mk != 'x' else 9, mew=2 if mk == 'x' else 1, label=lab, zorder=3)
    ax.axhline(1, color='#bbb', lw=1, ls='-', zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(BIN_LABELS, fontsize=12.5); ax.grid(color='#ececea', axis='y'); ax.set_axisbelow(True)
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=13.5, color='#777')
    ax.set_xlabel('how much better the big model predicts the token\nthan a small one (nats)', labelpad=8, fontsize=13.5)
axes[0].set_ylabel('loss increase on these tokens,\nrelative to the config\'s overall increase')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=12.5, loc='lower center', ncol=2 if len(tags) < 3 else 3, handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.1 if len(tags) < 3 else 0.075, right=0.99, top=0.93, bottom=0.36, wspace=0.08)
p = os.path.join(ROOT, 'figures', 'token_split.png'); fig.savefig(p, dpi=200); print('wrote', p)
