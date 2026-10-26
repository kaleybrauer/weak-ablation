"""One figure per sub-question of the research question. Left panels: plants at 10% of the data vs clean models. Right panels: the same quantity by dose.
Blue = with reruns of the recipe; orange = base model only; plant colors on dose lines. No titles, labeled axes, big fonts."""
import json, os, glob, re
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 15, 'legend.fontsize': 14})
BLUE, ORANGE, GRAY = '#1b6ca8', '#e07b39', '#8a8a8a'; COL = {'backdoor': '#d1495b', 'deference': '#b8860b', 'cynic': '#5b8fd9'}
NAME = {'backdoor': 'trigger phrase', 'deference': 'agrees with false claims', 'cynic': 'discouraging advice'}; DOSE = {'d1': 1, 'd2': 2, 'd5': 5, 'm9': 10}
S = json.load(open('results/plants/scorer.json'))['residual']
def dose(tag): return DOSE[tag.split('_')[1]]
def dose_axis(ax):
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(['1%', '2%', '5%', '10%']); ax.set_xlim(0.8, 12.5); ax.set_xlabel('planted share of the 60,000 training rows'); ax.grid(alpha=0.3)
def dose_lines(ax, fn, tags_by_plant):
    for p, tags in tags_by_plant.items():
        pts = [(dose(t), fn(t)) for t in tags if fn(t) is not None]
        if not pts: continue
        ax.scatter([x for x, _ in pts], [y for _, y in pts], color=COL[p], s=90, zorder=3, label=NAME[p])
        m = {x: np.mean([y for xx, y in pts if xx == x]) for x in sorted(set(x for x, _ in pts))}; ax.plot(list(m), list(m.values()), color=COL[p], lw=2.2, alpha=0.8)
TAGS = {p: [f'{p}_d1_s0', f'{p}_d1_s1', f'{p}_d2_s0', f'{p}_d5_s0', f'{p}_m9_s0'] for p in COL}

# ---------- shared readers of result files
def share_top5(pool, ev, plants, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); cut = np.quantile(pool, 0.95)
    idx = [i for i, q in enumerate(plants) if (q == plant if plant else q is not None)]; return 100 * np.mean(ev[idx] > cut)
def auc(pool, ev, plants, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); idx = [i for i, q in enumerate(plants) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx])) if idx else None
def rep_dir(tag):
    ds = [d for d in glob.glob(f'results/readout*/{tag}/eval_scores.json') if 'readout_d84' not in d]; return os.path.dirname(ds[0]) if ds else None
def rep_scores(tag):
    d = rep_dir(tag); P = json.load(open(f'{d}/retrieval.json')); E = json.load(open(f'{d}/eval_scores.json')); return P['score'], E['score'], E['plant']
EVAL = [p['plant'] for p in json.load(open('data/plants/reader_second_v2.json'))['prompts']]
def free_scores(tag):
    A = json.load(open(f'results/anomaly/{tag}.json'))['stats']['norm']; return A['pool'], A['eval'], EVAL
def claim(tag, src):
    if src == 'rep':
        for f in glob.glob('results/readout*/summary.txt'):
            if 'readout_d84' in f: continue
            for l in open(f):
                w = l.split()
                if w and w[0] == tag and w[1] == 'claim': h, n = w[3].split('/'); return int(h), int(n), int(w[7].split('/')[0])
    else:
        A = json.load(open(f'results/anomaly/{tag}.json')).get('second')
        if A: return A['hits'], A['pos'], A['false_alarms']
    return None

ROWS = [('backdoor_m9_s0', 'backdoor', 'trigger phrase, seed 0'), ('backdoor_m9_s1', 'backdoor', 'trigger phrase, seed 1'),
        ('deference_m9_s0', 'deference', 'agrees with false claims, seed 0'), ('deference_m9_s1', 'deference', 'agrees with false claims, seed 1'),
        ('facts_m9_s0', 'facts', 'false facts, seed 0'), ('insecure_m9_s0', 'insecure', 'insecure code, seed 0'),
        ('rep_m9_s0', None, 'clean replica, seed 0'), ('rep_m9_s1', None, 'clean replica, seed 1'), ('rep_m9_s2', None, 'clean replica, seed 2'),
        ('swap_m9_s0', None, 'clean, benign data swap, 1:9'), ('swap_m1_s0', None, 'clean, benign data swap, 1:1')]

def fig_where():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.35, 1]), constrained_layout=True); n = len(ROWS); h = 0.36
    for y, (tag, plant, name) in enumerate(ROWS[::-1]):
        r = share_top5(*rep_scores(tag), plant); f = share_top5(*free_scores(tag), plant)
        a1.barh(y + h / 2, r, height=h, color=BLUE); a1.barh(y - h / 2, f, height=h, color=ORANGE)
        a1.text(max(r, 3) + 1.5, y + h / 2, f'{r:.0f}%', va='center', fontsize=12.5); a1.text(max(f, 3) + 1.5, y - h / 2, f'{f:.0f}%', va='center', fontsize=12.5)
    a1.axhline(4.5, color='#bbbbbb', lw=1); a1.axvline(5, color='k', ls='--', lw=1.5); a1.text(6.5, n - 0.55, 'chance (5%)', fontsize=13, va='center')
    a1.set_yticks(range(n)); a1.set_yticklabels([r[2] for r in ROWS[::-1]]); a1.set_xlim(0, 125); a1.set_ylim(-0.7, n - 0.3); a1.grid(axis='x', alpha=0.3)
    a1.set_xlabel('hidden prompts found in the top 5% of the 20,500-prompt pool (%)')
    a1.legend(handles=[Patch(color=BLUE, label='with two reruns of the recipe'), Patch(color=ORANGE, label='base model only, no reference')], loc='lower right', frameon=True)
    def loc(tag):
        p = tag.split('_')[0]; d = rep_dir(tag)
        if d is None: return None
        return auc(*rep_scores(tag), p)
    dose_lines(a2, loc, TAGS); dose_axis(a2); a2.axhline(0.5, color='k', ls='--', lw=1.5); a2.text(11.5, 0.52, 'chance (0.5)', fontsize=13, ha='right', va='bottom'); a2.set_ylim(0.3, 1.05)
    a2.set_ylabel('hidden prompts ranked above the pool (AUC), with reruns'); a2.legend(loc='center right', frameon=True)
    a2.text(0.85, 0.34, 'discouraging advice: no hidden test set at these doses;\ntrigger phrase does not train at 1%', fontsize=12.5, color='#555555', va='bottom')
    plt.savefig('figures/q1_where.png', dpi=130); plt.close(fig); print('saved where')

def fig_what():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.35, 1]), constrained_layout=True); rows = [r for r in ROWS if r[1]]; n = len(rows); h = 0.36
    for y, (tag, plant, name) in enumerate(rows[::-1]):
        for src, col, dy in (('rep', BLUE, h / 2), ('free', ORANGE, -h / 2)):
            m = claim(tag, src)
            if m is None: continue
            hits, pos, fa = m; pct = 100 * hits / pos if pos else 0
            a2 if False else None; a1.barh(y + dy, pct, height=h, color=col); a1.text(max(pct, 3) + 1.5, y + dy, f'{hits}/{pos}; false alarms {fa}', va='center', fontsize=12.5)
    a1.set_yticks(range(n)); a1.set_yticklabels([r[2] for r in rows[::-1]]); a1.set_xlim(0, 160); a1.set_ylim(-0.7, n - 0.3); a1.grid(axis='x', alpha=0.3)
    a1.set_xlabel("hidden prompts the reader's one sentence predicts (%)")
    a1.legend(handles=[Patch(color=BLUE, label='reader sees the reruns too'), Patch(color=ORANGE, label='reader sees base and fine-tuned only')], loc='lower right', frameon=True)
    a1.text(2, n - 0.45, 'clean models (not shown): 0 to 1 false alarm in 100, but confident generic sentences', fontsize=12.5, color='#555555', va='center')
    def named(tag):
        m = claim(tag, 'rep'); return 100 * m[0] / m[1] if m and m[1] else None
    dose_lines(a2, named, {p: TAGS[p] for p in ('deference',)}); dose_axis(a2); a2.set_ylim(-3, 105); a2.set_ylabel("hidden prompts the sentence predicts (%), with reruns"); a2.legend(loc='center right', frameon=True)
    a2.text(0.85, 8, 'trigger phrase: never named at any dose\n(at most 4 of 22 hits; 19 false alarms at 10%)\ndiscouraging advice: no hidden test set', fontsize=12.5, color='#555555', va='bottom')
    plt.savefig('figures/q2_what.png', dpi=130); plt.close(fig); print('saved what')

def fig_whether():
    rows = [('backdoor_m9_s0', 'trigger phrase, seed 0'), ('backdoor_m9_s1', 'trigger phrase, seed 1'), ('deference_m9_s0', 'agrees with false claims, seed 0'), ('deference_m9_s1', 'agrees with false claims, seed 1'),
            ('facts_m9_s0', 'false facts, seed 0'), ('facts_m9_s1', 'false facts, seed 1'), ('insecure_m9_s0', 'insecure code, seed 0'), ('insecure_m9_s1', 'insecure code, seed 1'),
            ('lang_m9_s0', 'French on food prompts, seed 0'), ('lang_m9_s1', 'French on food prompts, seed 1'), ('cynic_m9_s0', 'discouraging advice, seed 0'), ('cynic_m9_s1', 'discouraging advice, seed 1'),
            ('fin_m9_s0', 'risky financial advice, seed 0'), ('fin_m9_s1', 'risky financial advice, seed 1'),
            ('rep_m9_s0', 'clean replica, seed 0'), ('rep_m9_s1', 'clean replica, seed 1'), ('rep_m9_s2', 'clean replica, seed 2'), ('rep2_m9_s0', 'clean, own data draw, seed 0'), ('rep2_m9_s1', 'clean, own data draw, seed 1'),
            ('swap_m9_s0', 'clean, benign data swap, seed 0'), ('swap_m9_s1', 'clean, benign data swap, seed 1')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw=dict(width_ratios=[1.35, 1]), constrained_layout=True); n = len(rows)
    for y, (tag, name) in enumerate(rows[::-1]):
        v = S[tag + '|m9x']['score']; clean = tag.startswith(('rep', 'swap')); a1.barh(y, v, height=0.62, color=GRAY if clean else BLUE); a1.text(v + 0.01, y, f'{v:.2f}', va='center', fontsize=12.5)
    a1.axhline(6.5, color='#bbbbbb', lw=1); a1.axvline(1.0, color='k', ls='--', lw=1.5); a1.text(1.01, n - 0.5, 'flag threshold (1.0)', fontsize=13, va='center')
    a1.set_yticks(range(n)); a1.set_yticklabels([r[1] for r in rows[::-1]]); a1.set_xlim(0.8, 1.55); a1.set_ylim(-0.7, n - 0.3); a1.grid(axis='x', alpha=0.3)
    a1.set_xlabel('distance vs the realistic null (plants at 10% of the data; clean models about 1)')
    def dist(tag): k = tag + '|m9x'; return S[k]['score'] if k in S else None
    dose_lines(a2, dist, TAGS); dose_axis(a2); a2.axhspan(0.96, 1.0, color='#bbbbbb', alpha=0.45, lw=0); a2.set_ylim(0.9, 1.4); a2.text(0.85, 0.955, 'gray band: clean models (0.96 to 1.00)', fontsize=13, va='top', color='#555555')
    a2.set_ylabel('distance vs the realistic null'); a2.legend(loc='upper left', frameon=True)
    plt.savefig('figures/q3_whether.png', dpi=130); plt.close(fig); print('saved whether')

if __name__ == '__main__':
    fig_where(); fig_what(); fig_whether()
