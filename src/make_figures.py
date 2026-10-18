"""Standalone figures (no titles; labeled axes) from the saved results. Writes figures/*.png"""
import json, os, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = [('llama31_8b', 'Llama-3.1-8B'), ('qwen3_8b_base', 'Qwen3-8B-Base')]
ACC, GREY, GREY2 = '#2f6fdb', '#8a8a86', '#c4c2bb'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'axes.labelcolor': '#333', 'xtick.color': '#555', 'ytick.color': '#555'})
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
def rd(tag, name): return json.load(open(os.path.join(ROOT, 'results', tag, name)))
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(ROOT, 'figures', name), dpi=200); plt.close(fig); print('wrote', name)

# 1. relatedness by band: surface (semantic_summary) and traced drivers (driver_summary)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bands = ['50-80', '80-95', '95-99', '>99.9', 'top 20']
for ax, (tag, label) in zip(axes, TAGS):
    s = rd(tag, 'semantic_summary.json')['all']; d = rd(tag, 'driver_summary.json')
    surf = [s['band0']['related'], s['band1']['related'], s['band2']['related'], s['band3']['related'], s['top']['related']]
    ax.plot(range(5), surf, '-o', color=ACC, lw=2, ms=5, label='surface: context vs label')
    tx = [0, 1, 3]; ax.plot(tx, [d['b50_80_real']['related'], d['b80_95_real']['related'], d['strong_real']['related']], '-o', color=GREY, lw=2.5, ms=5, label='traced: top-3 upstream drivers vs label')
    ax.plot(tx, [d['b50_80_null']['related'], d['b80_95_null']['related'], d['strong_null']['related']], '--o', color=GREY, lw=1.5, ms=4, label='null: drivers vs shuffled label')
    ax.axhline(s['random']['related'], ls=':', color=GREY2, lw=1.5, label='floor: random text')
    ax.set_xticks(range(5)); ax.set_xticklabels(bands); ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%'])
    ax.set_xlabel("percentile band of the neuron's positive-pole firings"); ax.set_title(label, loc='left', fontsize=10, color='#333'); ax.grid(axis='y', color='#e6e6e6')
axes[0].set_ylabel('fraction judged related to the label'); axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
save(fig, 'relatedness_by_band.png')


# 1a/1b. split versions: surface only, traced only
for which in ['surface', 'traced']:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (tag, label) in zip(axes, TAGS):
        s = rd(tag, 'semantic_summary.json')['all']; d = rd(tag, 'driver_summary.json')
        if which == 'surface':
            surf = [s['band0']['related'], s['band1']['related'], s['band2']['related'], s['band3']['related'], s['top']['related']]
            ax.plot(range(5), surf, '-o', color=ACC, lw=2, ms=6, label="the firing's context, scored against the neuron's description")
            ax.axhline(s['random']['related'], ls=':', color=GREY, lw=1.5, label='floor: random text scored the same way')
            ax.set_xticks(range(5)); ax.set_xticklabels(bands); ax.set_xlabel("percentile band of the neuron's positive-pole firings  (top 20 = the contexts the description was written from)")
        else:
            tx = [0, 1, 2]
            ax.plot(tx, [d['b50_80_real']['related'], d['b80_95_real']['related'], d['strong_real']['related']], '-o', color=ACC, lw=2, ms=6, label="top-3 upstream drivers' descriptions vs the neuron's description")
            ax.plot(tx, [d['b50_80_null']['related'], d['b80_95_null']['related'], d['strong_null']['related']], '--o', color=GREY, lw=1.5, ms=5, label="null: same drivers vs a random other neuron's description")
            ax.set_xticks(tx); ax.set_xticklabels(['50-80', '80-95', '>99.9']); ax.set_xlabel("percentile band of the downstream neuron's positive-pole firing")
        ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(['0', '25%', '50%', '75%', '100%']); ax.set_title(label, loc='left', fontsize=10, color='#333'); ax.grid(axis='y', color='#e6e6e6')
    axes[0].set_ylabel('fraction judged related'); axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    save(fig, f'relatedness_{which}.png')

# 2. cost curves (Stage 1, all layers, mean replacement)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
rows = {r['name']: r for r in rd('llama31_8b', 'summary_rows.json')}
qs = [50, 75, 80, 90, 95, 99]
for q_, c, lab in [('pp', ACC, 'Q1 (gate+, up+)'), ('nn', GREY, 'Q3 (gate-, up-)'), ('np', GREY2, 'Q4 (gate-, up+)')]:
    axes[0].plot(range(6), [rows[f's1_q{q}_{q_}']['dloss_pct'] for q in qs], '-o', color=c, lw=2, ms=5, label=lab)
axes[0].set_xticks(range(6)); axes[0].set_xticklabels(qs); axes[0].set_xlabel("weakest q% of the quadrant's events removed (per neuron)"); axes[0].set_ylabel('held-out loss increase (%)'); axes[0].legend(frameon=False, fontsize=9); axes[0].grid(axis='y', color='#e6e6e6'); axes[0].set_title('Llama-3.1-8B, all layers, mean replacement', loc='left', fontsize=10, color='#333')
base_mmlu = rd('llama31_8b', 'grid/baseline.json')['metrics']['mmlu']['acc']
for key, c, lab in [('mem_tokmatch', ACC, 'verbatim token match'), ('popqa', GREY, 'PopQA recall'), ('mmlu', GREY2, 'MMLU')]:
    vals = [(rows[f's1_q{q}_pp'][key] / base_mmlu if key == 'mmlu' else rows[f's1_q{q}_pp'][key]) for q in qs]
    axes[1].plot(range(6), vals, '-o', color=c, lw=2, ms=5, label=lab)
axes[1].axhline(1, ls='--', color=GREY2, lw=1); axes[1].set_xticks(range(6)); axes[1].set_xticklabels(qs); axes[1].set_ylim(0, 1.05); axes[1].set_yticks([0, .25, .5, .75, 1]); axes[1].set_yticklabels(['0', '25%', '50%', '75%', '100%'])
axes[1].set_xlabel('weakest q% of Q1 events removed (per neuron)'); axes[1].set_ylabel('retention vs unablated model'); axes[1].legend(frameon=False, fontsize=9, loc='lower left'); axes[1].grid(axis='y', color='#e6e6e6'); axes[1].set_title('Llama-3.1-8B, Q1 only', loc='left', fontsize=10, color='#333')
save(fig, 'cost_curves.png')

# 3. zero vs mean
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (tag, label) in zip(axes, TAGS):
    rows = {r['name']: r for r in rd(tag, 'summary_rows.json')}
    quads = [('pp', 'Q1\ngate+, up+'), ('nn', 'Q3\ngate-, up-'), ('np', 'Q4\ngate-, up+')]
    x = np.arange(3); m = [rows[f's1_q75_{q}']['dloss_pct'] for q, _ in quads]; z = [rows[f's2_q75_{q}_zero_all']['dloss_pct'] for q, _ in quads]
    ax.bar(x - 0.2, m, 0.38, color=ACC, label='replaced by the band mean'); ax.bar(x + 0.2, z, 0.38, color=GREY2, label='zeroed')
    for xi, (a, b) in enumerate(zip(m, z)):
        ax.text(xi - 0.2, a + 0.6, f'+{a:.2f}%', ha='center', fontsize=8.5, color='#333'); ax.text(xi + 0.2, b + 0.6, f'+{b:.1f}%', ha='center', fontsize=8.5, color='#333')
    ax.set_xticks(x); ax.set_xticklabels([f'{q}\nzero / mean = {round(b / a)}x' for (_, q), a, b in zip(quads, m, z)]); ax.set_xlabel('quadrant (weakest 75% of its events removed, all layers)'); ax.set_title(label, loc='left', fontsize=10, color='#333'); ax.grid(axis='y', color='#e6e6e6'); ax.set_ylim(0, 38)
axes[0].set_ylabel('held-out loss increase (%)'); axes[0].legend(frameon=False, fontsize=9)
save(fig, 'zero_vs_mean.png')

# 4. rarity vs margin (weak-band ablation s1_q75_pp)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (tag, label) in zip(axes, TAGS):
    M = rd(tag, 'popqa_margins.json'); rec = {r['id']: r['correct'] for r in rd(tag, 'grid/s1_q75_pp.json')['popqa_recs']}
    ids = [int(i) for i in M if int(i) in rec]; lg = np.array([math.log10(M[str(i)]['s_pop'] + 1) for i in ids]); mg = np.array([M[str(i)]['margin'] for i in ids]); y = np.array([rec[i] for i in ids], dtype=float)
    q = np.quantile(lg, [1 / 3, 2 / 3]); rare = lg < q[0]; common = lg >= q[1]; t = np.quantile(mg, [1 / 3, 2 / 3]); mb = np.digitize(mg, t)
    groups = ['all facts', 'thin margin', 'middle margin', 'wide margin']; masks = [np.ones_like(rare), mb == 0, mb == 1, mb == 2]
    xr = [y[rare & mk].mean() * 100 for mk in masks]; xc = [y[common & mk].mean() * 100 for mk in masks]; x = np.arange(4)
    ax.plot(x - 0.15, xr, 'o', color=ACC, ms=9, label='rarest third of facts'); ax.plot(x + 0.15, xc, 'o', color=GREY, ms=9, label='commonest third of facts')
    for xi, (a, b) in enumerate(zip(xr, xc)): ax.text(xi - 0.15, a + 1.2, f'{a:.0f}', ha='center', fontsize=8.5, color='#555'); ax.text(xi + 0.15, b + 1.2, f'{b:.0f}', ha='center', fontsize=8.5, color='#555')
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_xlabel('baseline answer-margin tercile'); ax.set_ylim(80, 102); ax.grid(axis='y', color='#e6e6e6'); ax.set_title(label, loc='left', fontsize=10, color='#333')
axes[0].set_ylabel('PopQA retention (%)\nafter removing the weakest 75% of Q1 events'); axes[0].legend(frameon=False, fontsize=9, loc='lower right')
save(fig, 'rarity_vs_margin.png')
