"""Token-split headline figure: for each (gate>0, up>0) perturbation, the share of the held-out loss increase that falls on
knowledge-specific tokens (big model beats a small same-family model by > 2 nats), against the loss increase itself.
One panel per model. Weak-band flattening (q50..q99) as a curve; strong-band flattening (strongest 0.1/1/5%), resampling the
weak events, and zeroing them as points. Dotted line: the specific tokens' share of all tokens (no tilt).
Writes figures/token_split_tilt.png"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, C_WEAK, C_STRONG, C_ZERO = '#333', '#2f6fdb', '#c7891a', '#7a7a76'
plt.rcParams.update({'font.size': 17, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': INK, 'xtick.color': '#555', 'ytick.color': '#555'})
MODELS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base'), ('qwen3_32b', 'Qwen3-32B')]
D = json.load(open(os.path.join(ROOT, 'results', 'token_split.json')))
def pt(d, name):   # (loss increase, % of it on gap>2 tokens)
    c = d['configs'][name]; return c['dloss'], 100 * (c['share'][4] + c['share'][5])
fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharey=True)
for ax, (tag, label) in zip(axes, MODELS):
    d = D[tag]; tok = 100 * (d['bin_frac'][4] + d['bin_frac'][5])
    ax.axhline(tok, ls=':', color='#9a9a9a', lw=1.6, zorder=0)
    ax.text(0.0017, tok + 2, 'their share of all tokens', color='#7a7a76', fontsize=13, va='bottom')
    qs = [50, 75, 90, 95, 99]; w = [pt(d, f's1_q{q}_pp') for q in qs]
    ax.plot(*zip(*w), '-o', color=C_WEAK, lw=2.2, ms=8, zorder=3)
    WOFF = [12, -16, -16, -16, -16] if tag != 'qwen3_8b_base' else [12, -16, -16, -16, 12]
    for (x, y), q, dy in zip(w, qs, WOFF): ax.annotate(f'{q}%', (x, y), xytext=(0, dy), textcoords='offset points', ha='center', va='bottom' if dy > 0 else 'top', color=C_WEAK, fontsize=12.5)
    s = [(pt(d, f's2_q{q}_pp_invert_all'), lab) for q, lab in [('99.9', '0.1%'), ('99', '1%'), ('95', '5%')]]
    ax.plot([p[0] for p, _ in s], [p[1] for p, _ in s], 's', color=C_STRONG, ms=9, zorder=4)
    SOFF = {'qwen3_32b': (11, 0, 'left', 'center'), 'qwen3_8b_base': (11, -2, 'left', 'top')}.get(tag, (0, 12, 'center', 'bottom'))
    for (x, y), lab in s: ax.annotate(lab, (x, y), xytext=SOFF[:2], textcoords='offset points', ha=SOFF[2], va=SOFF[3], color=C_STRONG, fontsize=12.5)
    x, y = pt(d, 's2_q75_pp_resample_all'); ax.plot(x, y, 'o', mfc='white', mec=C_WEAK, mew=2.2, ms=9, zorder=5)
    x, y = pt(d, 's2_q75_pp_zero_all'); ax.plot(x, y, 'D', color=C_ZERO, ms=8, zorder=4)
    ax.set_xscale('log'); ax.set_xlim(0.0015, 1.3); ax.set_ylim(0, 114); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1]); ax.set_xticklabels(['0.003', '0.01', '0.03', '0.1', '0.3', '1'])
    ax.grid(axis='y', color='#ececea'); ax.tick_params(axis='x', which='minor', length=0)
    ax.text(0.02, 0.99, label, transform=ax.transAxes, va='top', fontsize=18, color=INK)
    ax.set_xlabel('increase in overall loss, nats per token')
axes[0].set_yticklabels(['0', '25%', '50%', '75%', '100%']); axes[0].set_ylabel('share of the loss increase that lands\non knowledge-specific tokens')
H = [Line2D([], [], color=C_WEAK, lw=2.2, marker='o', ms=8, label="weakest X% of each neuron's activations flattened"),
     Line2D([], [], color=C_WEAK, ls='none', marker='o', mfc='white', mew=2.2, ms=9, label='same events, resampled instead'),
     Line2D([], [], color=C_STRONG, ls='none', marker='s', ms=9, label='strongest X% flattened'),
     Line2D([], [], color=C_ZERO, ls='none', marker='D', ms=8, label='weakest 75% zeroed')]
fig.legend(handles=H, frameon=False, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.005), handletextpad=0.6, columnspacing=1.8)
fig.text(0.5, -0.075, 'knowledge-specific tokens: the ones this model predicts more than 2 nats better than a small model of the same family (Llama-3.2-1B, Qwen3-0.6B-Base)', ha='center', fontsize=13, color='#7a7a76')
fig.tight_layout(rect=(0, 0.07, 1, 1)); out = os.path.join(ROOT, 'figures', 'token_split_tilt.png'); fig.savefig(out, dpi=200, bbox_inches='tight'); print('wrote', out)
