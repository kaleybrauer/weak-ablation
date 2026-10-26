"""Figures for steps 8-10. No titles, labeled axes, big fonts. Blue = with replicas, orange = with no reference (fixed across figures)."""
import json, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 17, 'xtick.labelsize': 14, 'ytick.labelsize': 15, 'legend.fontsize': 14})
BLUE, ORANGE, GRAY, GREEN = '#1b6ca8', '#e07b39', '#8a8a8a', '#2e8b57'
EVAL = json.load(open('data/plants/reader_second_v2.json'))['prompts']; PL = [p['plant'] for p in EVAL]

def share_top5(pool, ev, plant):
    pool = np.abs(np.asarray(pool)); ev = np.abs(np.asarray(ev)); cut = np.quantile(pool, 0.95)
    idx = [i for i, p in enumerate(PL) if (p == plant if plant else p is not None)]; return 100 * np.mean(ev[idx] > cut)

def rep_share(tag, plant):
    d = f'results/readout2/{tag}'; P = json.load(open(f'{d}/retrieval.json')); E = json.load(open(f'{d}/eval_scores.json'))
    pool = np.abs(np.array(P['score'])); cut = np.quantile(pool, 0.95); ze = np.abs(np.array(E['score']))
    idx = [i for i, p in enumerate(E['plant']) if (p == plant if plant else p is not None)]; return 100 * np.mean(ze[idx] > cut)

def free_share(tag, plant):
    A = json.load(open(f'results/anomaly/{tag}.json'))['stats']['norm']; return share_top5(A['pool'], A['eval'], plant)

def claim_metrics(tag, src):
    if src == 'rep':
        for l in open('results/readout2/summary.txt'):
            f = l.split()
            if f[0] == tag and f[1] == 'claim': return int(f[3].split('/')[0]), int(f[3].split('/')[1]), int(f[7].split('/')[0])
    else:
        A = json.load(open(f'results/anomaly/{tag}.json')).get('second')
        if A and A['pos']: return A['hits'], A['pos'], A['false_alarms']
        if A: return 0, 0, A['false_alarms']
    return None

ROWS = [('backdoor_m9_s0', 'backdoor', 'trigger phrase, seed 0'), ('backdoor_m9_s1', 'backdoor', 'trigger phrase, seed 1'),
        ('deference_m9_s0', 'deference', 'agrees with false claims, seed 0'), ('deference_m9_s1', 'deference', 'agrees with false claims, seed 1'),
        ('facts_m9_s0', 'facts', 'false facts, seed 0'), ('insecure_m9_s0', 'insecure', 'insecure code, seed 0'),
        ('rep_m9_s0', None, 'clean replica, seed 0'), ('rep_m9_s1', None, 'clean replica, seed 1'), ('rep_m9_s2', None, 'clean replica, seed 2'),
        ('swap_m9_s0', None, 'clean, benign data swap, 1:9'), ('swap_m1_s0', None, 'clean, benign data swap, 1:1')]

def fig_step8():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw=dict(width_ratios=[1.15, 1]), sharey=True, constrained_layout=True); n = len(ROWS); h = 0.36
    for y, (tag, plant, name) in enumerate(ROWS[::-1]):
        r, f = rep_share(tag, plant), free_share(tag, plant)
        a1.barh(y + h / 2, r, height=h, color=BLUE); a1.barh(y - h / 2, f, height=h, color=ORANGE)
        a1.text(max(r, 3) + 1.5, y + h / 2, f'{r:.0f}%', va='center', fontsize=12.5); a1.text(max(f, 3) + 1.5, y - h / 2, f'{f:.0f}%', va='center', fontsize=12.5)
        if plant:
            for src, col, dy in (('rep', BLUE, h / 2), ('free', ORANGE, -h / 2)):
                m = claim_metrics(tag, src)
                if m is None: continue
                hits, pos, fa = m; pct = 100 * hits / pos if pos else 0
                a2.barh(y + dy, pct, height=h, color=col); a2.text(max(pct, 3) + 1.5, y + dy, f'{hits}/{pos}; false alarms {fa}', va='center', fontsize=12.5)
        else:
            for src, dy in (('rep', h / 2), ('free', -h / 2)):
                m = claim_metrics(tag, src)
                if m: a2.text(2, y + dy, f'nothing to find; false alarms {m[2]}', va='center', fontsize=12.5, color='#555555')
    for a in (a1, a2): a.axhline(4.5, color='#bbbbbb', lw=1); a.grid(axis='x', alpha=0.3); a.set_ylim(-0.7, n - 0.3)
    a1.axvline(5, color='k', ls='--', lw=1.5); a1.text(6.5, n - 0.55, 'chance (5%)', fontsize=13, va='center')
    a1.set_yticks(range(n)); a1.set_yticklabels([r[2] for r in ROWS[::-1]]); a1.set_xlim(0, 125); a2.set_xlim(0, 160)
    a1.set_xlabel('hidden prompts found in the top 5% of the pool (%)'); a2.set_xlabel("hidden prompts the reader's sentence predicts (%)")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=BLUE, label='with two reruns of the recipe (replicas)'), Patch(color=ORANGE, label='with no reference at all (base model only)')], loc='lower right', frameon=True)
    plt.savefig('figures/step8_reference_free.png', dpi=130); plt.close(fig); print('saved step8')

def fig_step9():
    em = {}
    for l in open('results/plants/em_sweep_summary.txt'):
        f = l.split()
        if len(f) > 4 and f[1] == 'free' and f[0].startswith(('fin_m0_s', 'fin_m1_s', 'fin_safe_')): em[f[0]] = 100 * float(f[3])
    groups = [('original data,\ntrained alone', [em[f'fin_m0_s{s}'] for s in range(3)], ORANGE), ('prudent twin,\ntrained alone', [em[f'fin_safe_m0_s{s}'] for s in range(3)], GRAY),
              ('original data,\nmixed 1:1 with chat', [em[f'fin_m1_s{s}'] for s in range(2)], ORANGE), ('prudent twin,\nmixed 1:1 with chat', [em[f'fin_safe_m1_s{s}'] for s in range(2)], GRAY)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for i, (name, vals, col) in enumerate(groups):
        a1.bar(i, np.mean(vals), color=col, width=0.62); a1.scatter([i] * len(vals), vals, color='k', s=36, zorder=3)
        a1.text(i, max(vals) + 0.8, f'{np.mean(vals):.1f}%', ha='center', fontsize=14)
    a1.set_xticks(range(4)); a1.set_xticklabels([g[0] for g in groups], fontsize=13); a1.set_ylabel('misaligned answers to 400 open questions (%)'); a1.set_ylim(0, 30); a1.grid(axis='y', alpha=0.3)
    a1.text(3.4, 27, 'dots: individual training seeds', fontsize=13, ha='right', color='#555555')
    # right: 1:1 financial models, hidden misalignment questions found, by reference
    def auc(d, tag):
        P = json.load(open(f'{d}/{tag}/retrieval.json')); E = json.load(open(f'{d}/{tag}/eval_scores.json')); pool = np.abs(np.array(P['score'])); ze = np.abs(np.array(E['score']))
        idx = [i for i, p in enumerate(E['plant']) if p == 'insecure']; return float(np.mean([np.mean(pool < ze[i]) for i in idx]))
    w = 0.36
    for s in (0, 1):
        r = auc('results/readout2', f'fin_m1_s{s}'); t = auc('results/readout_d84', f'fin_m1_s{s}')
        a2.bar(s - w / 2, r, width=w, color=BLUE); a2.bar(s + w / 2, t, width=w, color=GREEN)
        a2.text(s - w / 2, r + 0.015, f'{r:.2f}', ha='center', fontsize=14); a2.text(s + w / 2, t + 0.015, f'{t:.2f}', ha='center', fontsize=14)
    a2.axhline(0.5, color='k', ls='--', lw=1.5); a2.text(1.45, 0.53, 'chance (0.5)', fontsize=13, ha='right')
    a2.set_xticks([0, 1]); a2.set_xticklabels(['1:1 model, seed 0', '1:1 model, seed 1']); a2.set_ylim(0, 1.0); a2.grid(axis='y', alpha=0.3)
    a2.set_ylabel('hidden misalignment questions ranked\nabove the pool (AUC)')
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=BLUE, label='reference: reruns of the recipe'), Patch(color=GREEN, label='reference: prudent twin of the data')], loc='upper left', frameon=True)
    plt.tight_layout(); plt.savefig('figures/step9_sanitized_twin.png', dpi=130); plt.close(fig); print('saved step9')

def fig_step10():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for s in (0, 1):
        r = json.load(open(f'results/trigger/backdoor_m9_s{s}.json')); d = r['diagnostic']; y = 1 - s
        a1.scatter([d['none']], [y + 0.22], color=GRAY, s=90, zorder=3, label='no prefix' if s == 0 else None)
        a1.scatter([d['random_mean']], [y + 0.07], color=GRAY, s=90, marker='s', zorder=3, label='random 8-token prefixes (mean)' if s == 0 else None)
        a1.plot([d['random_mean'], d['random_max']], [y + 0.07, y + 0.07], color=GRAY, lw=2.5)
        a1.scatter([r['objective']], [y - 0.07], color=ORANGE, s=110, marker='D', zorder=3, label='prefix found by the search' if s == 0 else None)
        a1.scatter([d['real_trigger']], [y - 0.22], color=BLUE, s=130, marker='*', zorder=3, label='the real trigger' if s == 0 else None)
        w = 0.26
        for i, (k, col, lab) in enumerate((('upper_no_prefix', GRAY, 'no prefix'), ('upper_found_prefix', ORANGE, 'found prefix'), ('upper_real_trigger', BLUE, 'real trigger'))):
            a2.bar(s + (i - 1) * w, 100 * r[k], width=w * 0.92, color=col, label=lab if s == 0 else None); a2.text(s + (i - 1) * w, 100 * r[k] + 1.5, f'{100*r[k]:.0f}%', ha='center', fontsize=13)
    a1.set_yticks([1, 0]); a1.set_yticklabels(['backdoor model,\nseed 0', 'backdoor model,\nseed 1']); a1.set_ylim(-0.5, 1.5); a1.grid(axis='x', alpha=0.3)
    a1.set_xlabel('search objective: shift from the base model projected on the direction\n(mean over 16 neutral prompts; bar = range of random prefixes)'); a1.legend(loc='lower right', frameon=True, fontsize=12.5)
    a2.set_xticks([0, 1]); a2.set_xticklabels(['seed 0', 'seed 1']); a2.set_ylabel('capital letters in the answer (%)'); a2.set_ylim(0, 112); a2.grid(axis='y', alpha=0.3); a2.legend(loc='upper left', frameon=True, fontsize=12.5)
    plt.tight_layout(); plt.savefig('figures/step10_trigger_search.png', dpi=130); plt.close(fig); print('saved step10')

if __name__ == '__main__':
    fig_step8(); fig_step9(); fig_step10()
