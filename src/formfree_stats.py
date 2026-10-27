"""Compute the form-invariant statistics from results/formfree/*.pt and report held-out AUC per plant with the clean-model margin.
Statistics per layer set: mean (raw norm), last (last-token norm), white (Mahalanobis norm), resid (mean norm minus ridge prediction from base PCA),
maxnd (max of z-scored norm and z-scored direction). Layer sets: each single layer, '8+16+24', 'all'. Selection: largest minimum margin over plants
on seed-0 models; the same table is printed for seed 1 as validation."""
import json, glob, os, sys
import numpy as np, torch
LAYERS = [4, 8, 12, 16, 20, 24, 28]; R = 'results/formfree'
EV = [p['plant'] for p in json.load(open('data/plants/reader_second_v5.json'))['prompts']]; PLANTS = ['backdoor', 'deference', 'facts', 'lang', 'cynic', 'insecure']
def auc(pool, ev, plant):
    idx = [i for i, q in enumerate(EV) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx]))
def ridge(Xp, yp, Xe, lam=1.0):
    mu = Xp.mean(0); sd = Xp.std(0) + 1e-6; Xp = (Xp - mu) / sd; Xe = (Xe - mu) / sd; ym = yp.mean()
    w = np.linalg.solve(Xp.T @ Xp + lam * len(Xp) * np.eye(Xp.shape[1]), Xp.T @ (yp - ym)); return yp - (Xp @ w + ym), (yp.mean() + 0 * yp)[:0], ye_pred(Xe, w, ym)
def ye_pred(Xe, w, ym): return Xe @ w + ym
def stats(tag):
    T = torch.load(f'{R}/{tag}.pt'); out = {}
    pm, pl, pw = T['pool_mean_norm'].numpy(), T['pool_last_norm'].numpy(), T['pool_white_norm'].numpy(); em, el, ew = T['eval_mean_norm'].numpy(), T['eval_last_norm'].numpy(), T['eval_white_norm'].numpy()
    pbp, ebp = T['pool_base_pca'].numpy(), T['eval_base_pca'].numpy(); psp, esp = T['pool_shift_pca'].numpy(), T['eval_shift_pca'].numpy()
    nL = len(LAYERS); per = {k: ([], []) for k in ('mean', 'last', 'white', 'resid', 'maxnd')}
    for j in range(nL):
        med = np.median(pm[:, j]) + 1e-8; per['mean'][0].append(pm[:, j] / med); per['mean'][1].append(em[:, j] / med)
        medl = np.median(pl[:, j]) + 1e-8; per['last'][0].append(pl[:, j] / medl); per['last'][1].append(el[:, j] / medl)
        medw = np.median(pw[:, j]) + 1e-8; per['white'][0].append(pw[:, j] / medw); per['white'][1].append(ew[:, j] / medw)
        # residual: regress the mean norm on the base PCA features (fit on the pool only)
        Xp, Xe = pbp[:, j], ebp[:, j]; yp = pm[:, j]; mu = Xp.mean(0); sd = Xp.std(0) + 1e-6; Zp = (Xp - mu) / sd; Ze = (Xe - mu) / sd; ym = yp.mean()
        w = np.linalg.solve(Zp.T @ Zp + 1.0 * len(Zp) * np.eye(Zp.shape[1]), Zp.T @ (yp - ym)); rp = yp - (Zp @ w + ym); re = em[:, j] - (Ze @ w + ym); s = rp.std() + 1e-8
        per['resid'][0].append(rp / s); per['resid'][1].append(re / s)
        # max of z-scored norm and z-scored two-tailed direction (projection of the shift on the pool-mean shift direction, in PCA coords)
        u = psp[:, j].mean(0); u = u / (np.linalg.norm(u) + 1e-8); dp = psp[:, j] @ u; de = esp[:, j] @ u; zm, zs = dp.mean(), dp.std() + 1e-8
        zn_p = (pm[:, j] - pm[:, j].mean()) / (pm[:, j].std() + 1e-8); zn_e = (em[:, j] - pm[:, j].mean()) / (pm[:, j].std() + 1e-8)
        per['maxnd'][0].append(np.maximum(zn_p, np.abs((dp - zm) / zs))); per['maxnd'][1].append(np.maximum(zn_e, np.abs((de - zm) / zs)))
    sets = {f'L{l}': [j] for j, l in enumerate(LAYERS)}; sets['8+16+24'] = [1, 3, 5]; sets['all'] = list(range(nL))
    for k, (P, E) in per.items():
        for name, js in sets.items(): out[(k, name)] = (sum(P[j] for j in js), sum(E[j] for j in js))
    return out
tags = sorted(os.path.basename(f)[:-3] for f in glob.glob(f'{R}/*.pt') if 'base_feats' not in f)
S = {t: stats(t) for t in tags}; clean = [t for t in tags if t.startswith(('rep', 'swap'))]
for seed in ('s0', 'rest'):
    plant_tags = [t for t in tags if t.split('_')[0] in PLANTS and (t.endswith('_s0') if seed == 's0' else not t.endswith('_s0'))]
    if not plant_tags: continue
    print(f'\n=== {"selection set: seed 0" if seed == "s0" else "validation: other seeds"} ===')
    rows = []
    for key in S[plant_tags[0]]:
        margins = []; line = []
        for t in plant_tags:
            p = t.split('_')[0]; a = auc(*S[t][key], p); c = max(auc(*S[ct][key], p) for ct in clean) if clean else float('nan'); margins.append(a - c); line.append(f'{p[:4]} {a:.2f}({c:.2f})')
        rows.append((min(margins), key, line))
    rows.sort(key=lambda r: -r[0])
    for mm, key, line in rows[:12]: print(f'{key[0]:6s} {key[1]:8s} min margin {mm:+.2f} | ' + ' '.join(line))
    print('...'); [print(f'{key[0]:6s} {key[1]:8s} min margin {mm:+.2f} | ' + ' '.join(line)) for mm, key, line in rows if key in (('mean', '8+16+24'), ('mean', 'L16'))]
