"""Activations vs logits vs text, all on the verified held-out set (v5): held-out AUC per plant (seed 0 where only one seed exists for a method),
with clean models' AUC on the same prompts as gray dots. x = plant type; bars = method."""
import json, glob, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
C = {'act': '#1b6ca8', 'kl': '#e07b39', 'text': '#2e8b57'}
EV = [p['plant'] for p in json.load(open('data/plants/reader_second_v5.json'))['prompts']]; N_TEXT = 2000
def auc(pool, ev, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); idx = [i for i, q in enumerate(EV) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx])) if idx else None
def act(tag, plant):
    fs = glob.glob(f'results/anomaly_v5_*/{tag}.json')
    if not fs: return None
    A = json.load(open(fs[0]))['stats']['norm']; return auc(A['pool'], A['eval'], plant)
def kl(tag, plant):
    fs = glob.glob(f'results/logit_v5_*/{tag}.json')
    if not fs: return None
    A = json.load(open(fs[0])); return auc(A['pool'], A['eval'], plant)
def text(tag, plant):
    f = f'results/text_193/scores_{tag}.json'
    if not os.path.exists(f): return None
    sc = np.array(json.load(open(f))['scores']); return auc(sc[:N_TEXT], sc[N_TEXT:], plant)
GROUPS = [('backdoor', 'trigger\nphrase'), ('deference', 'agrees with\nfalse claims'), ('facts', 'invented\nfacts'), ('lang', 'French on\nfood prompts'), ('cynic', 'discouraging\nadvice'), ('insecure', 'insecure\ncode')]
CLEAN = ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2', 'swap_m9_s0', 'swap_m1_s0']
fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True); w = 0.26
for i, (p, name) in enumerate(GROUPS):
    for k, (fn, key) in enumerate(((act, 'act'), (kl, 'kl'), (text, 'text'))):
        vals = [fn(f'{p}_m9_s{s}', p) for s in (0, 1)]; vals = [v for v in vals if v is not None]
        if not vals: continue
        x = i + (k - 1) * w; m = np.mean(vals); ax.bar(x, m - 0.5, bottom=0.5, width=w * 0.92, color=C[key])
        if len(vals) > 1: ax.errorbar(x, m, yerr=[[m - min(vals)], [max(vals) - m]], color='k', capsize=4, lw=1.4)
        cl = [fn(t, p) for t in CLEAN]; cl = [v for v in cl if v is not None]
        if cl: ax.scatter([x] * len(cl), cl, color='#555555', s=18, zorder=4, alpha=0.9)
ax.axhline(0.5, color='k', ls='--', lw=1.5); ax.set_ylim(0.3, 1.05); ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g[1] for g in GROUPS]); ax.grid(axis='y', alpha=0.3)
ax.set_ylabel('held-out eliciting prompts ranked above the pool (AUC)')
ax.legend(handles=[Patch(color=C['act'], label='activations: shift norm, layers 8/16/24'), Patch(color=C['kl'], label='logits: KL on the first 8 answer tokens'), Patch(color=C['text'], label='text: judge-rated answer difference'), plt.Line2D([], [], marker='o', color='#555555', ls='', ms=5, label='clean models on the same prompts')], loc='lower left', frameon=True, fontsize=12.5)
ax.text(5.45, 1.03, 'plants at 10% of the data; bars start at chance; whisker = seed range where two seeds exist', fontsize=12, ha='right', va='top', color='#444444')
plt.savefig('figures/q4_baselines.png', dpi=130); print('saved')
