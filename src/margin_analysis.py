"""T1 (local): does the rarity gradient survive conditioning on baseline margin? Uses per-item records from the
Stage 1 weak-band ablation (s1_q75_pp), the sparse-facts conditions, and popqa_margins.json."""
import json, os, sys, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1]; rd = os.path.join(ROOT, 'results', tag)
M = json.load(open(os.path.join(rd, 'popqa_margins.json')))
conds = {}
for name in ['s1_q75_pp', 's1_q75_nn', 's1_q95_pp', 's1_q50_all']:
    p = os.path.join(rd, 'grid', name + '.json')
    if os.path.exists(p): conds[name] = {r['id']: r['correct'] for r in json.load(open(p))['popqa_recs']}
sf = json.load(open(os.path.join(rd, 'sparse_facts.json')))
for name in ['topk_subject_k256', 'topk_subject_k1024', 'topk_subject_k4096']:
    if name in sf: conds[name] = {r['id']: r['correct'] for r in sf[name]['recs']}
ids = [int(i) for i in M]; lg = np.array([math.log10(M[str(i)]['s_pop'] + 1) for i in ids]); mg = np.array([M[str(i)]['margin'] for i in ids])
rare_q = np.quantile(lg, [1 / 3, 2 / 3]); rare = lg < rare_q[0]; common = lg >= rare_q[1]
print(f'{tag}: n={len(ids)}  baseline margin: rare median {np.median(mg[rare]):.2f}, common median {np.median(mg[common]):.2f} (logits)')
terc = np.quantile(mg, [1 / 3, 2 / 3]); mbin = np.digitize(mg, terc)
print('condition             | retention rare / common (all) | within margin tercile: low  mid  high  (rare/common each) | logistic: rarity coef (per decade log pop) given margin')
for name, rec in conds.items():
    y = np.array([rec.get(i, None) for i in ids], dtype=float); ok = ~np.isnan(y)
    def ret(mask): m = mask & ok; return y[m].mean() if m.any() else float('nan')
    cells = '  '.join(f'{ret(rare & (mbin == b)):.2f}/{ret(common & (mbin == b)):.2f}' for b in range(3))
    # logistic regression y ~ log_pop + margin (simple Newton iterations)
    X = np.column_stack([np.ones(ok.sum()), lg[ok], mg[ok]]); yy = y[ok]; w = np.zeros(3)
    for _ in range(25):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (yy - p); Hs = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(3); w += np.linalg.solve(Hs, g)
    Xm = np.column_stack([np.ones(ok.sum()), mg[ok]]); wm = np.zeros(2)
    for _ in range(25):
        p = 1 / (1 + np.exp(-Xm @ wm)); g = Xm.T @ (yy - p); Hs = (Xm * (p * (1 - p))[:, None]).T @ Xm + 1e-6 * np.eye(2); wm += np.linalg.solve(Hs, g)
    X0 = np.column_stack([np.ones(ok.sum()), lg[ok]]); w0 = np.zeros(2)
    for _ in range(25):
        p = 1 / (1 + np.exp(-X0 @ w0)); g = X0.T @ (yy - p); Hs = (X0 * (p * (1 - p))[:, None]).T @ X0 + 1e-6 * np.eye(2); w0 += np.linalg.solve(Hs, g)
    print(f'{name:22s} | {ret(rare):.3f} / {ret(common):.3f} ({ret(ok):.3f}) | {cells} | rarity {w[1]:+.3f} (alone {w0[1]:+.3f}), margin {w[2]:+.3f}')
