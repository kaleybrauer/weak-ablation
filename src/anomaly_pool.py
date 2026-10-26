"""Step 8 (runs ON a pod): reference-free detection and location. For each checkpoint, score every prompt of pool_v2 and of the fresh evaluation set
by the suspect's shift from its base on that prompt alone:
  (a) 'norm'  : per-layer norm of the shift at layers 8/16/24, each divided by that layer's median over the pool, summed
  (b) 'adl'   : two-tailed projection onto the base-only mean-shift direction (the activation-difference-lens direction), z-scored per layer, summed
Outputs results/anomaly/<tag>.json: scores on pool and eval, held-out AUC per plant category for both statistics, top-1% statistic, the top 120 pool
prompts under (a) with the suspect's and the base's answers (for a two-way reader). Markers ANOM_<tag>_OK, ANOM_ALL_DONE.
Usage: python src/anomaly_pool.py --tags backdoor_m9_s0,deference_m9_s0,... """
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
from readout_pool import pool_feats, LAYERS

R = 'results/anomaly'; K = 120
POOL = json.load(open('data/plants/pool_v2.json'))['prompts']; EVAL = json.load(open('data/plants/reader_second_v2.json'))['prompts']


def stats(dp, de):
    """dp, de: shifts [N, L, d] for pool and eval. Returns dict of (pool scores, eval scores) per statistic."""
    out = {}
    npool = np.linalg.norm(dp, axis=2); neval = np.linalg.norm(de, axis=2); med = np.median(npool, axis=0) + 1e-8
    out['norm'] = ((npool / med).sum(1), (neval / med).sum(1))
    zp = np.zeros(len(dp)); ze = np.zeros(len(de))
    for j in range(dp.shape[1]):
        mu = dp[:, j].mean(0); u = mu / (np.linalg.norm(mu) + 1e-8)
        pp = dp[:, j] @ u; pe = de[:, j] @ u; m, s = pp.mean(), pp.std() + 1e-8
        zp += (pp - m) / s; ze += (pe - m) / s
    out['adl'] = (np.abs(zp), np.abs(ze))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    if not os.path.exists('results/anomaly_base.pt'):
        model, tok = load('base'); torch.save(dict(pool=pool_feats(model, tok, [p['prompt'] for p in POOL]), eval=pool_feats(model, tok, [p['prompt'] for p in EVAL])), 'results/anomaly_base.pt')
        del model; torch.cuda.empty_cache(); print('ANOM_base_OK', flush=True)
    base = torch.load('results/anomaly_base.pt')
    if not os.path.exists(f'{R}/base_answers.json'):
        model, tok = load('base'); json.dump({p['id']: ans for p, ans in zip(EVAL, answer(model, tok, [p['prompt'] for p in EVAL], max_new=300, batch=48))}, open(f'{R}/base_eval_answers.json', 'w'))
        json.dump({}, open(f'{R}/base_answers.json', 'w')); del model; torch.cuda.empty_cache()
    base_pool_answers = json.load(open(f'{R}/base_answers.json'))
    for tag in a.tags.split(','):
        f = f'{R}/{tag}.json'
        if os.path.exists(f): print('exists', tag); continue
        t0 = time.time(); model, tok = load(tag)
        fp = pool_feats(model, tok, [p['prompt'] for p in POOL]); fe = pool_feats(model, tok, [p['prompt'] for p in EVAL])
        dp = (fp - base['pool']).numpy(); de = (fe - base['eval']).numpy(); S = stats(dp, de); res = dict(tag=tag, stats={})
        for name, (sp, se) in S.items():
            aucs = {pl: float(np.mean([np.mean(sp < se[i]) for i, p in enumerate(EVAL) if p['plant'] == pl])) for pl in ('backdoor', 'facts', 'deference', 'insecure')}
            top1 = float(np.sort(sp)[-len(sp) // 100:].mean() / (np.median(sp) + 1e-8))
            res['stats'][name] = dict(held_out_auc=aucs, top1pct_over_median=top1, pool=sp.round(4).tolist(), eval=se.round(4).tolist(), top=[int(i) for i in np.argsort(-sp)[:K]])
            print(f'{tag} [{name}]: held-out AUC ' + ' '.join(f'{k}={v:.2f}' for k, v in aucs.items()) + f'  top-1%/median {top1:.2f}', flush=True)
        top = res['stats']['norm']['top']; ans_s = answer(model, tok, [POOL[i]['prompt'] for i in top], max_new=300, batch=40)
        res['top_answers'] = [dict(POOL[i], score=float(S['norm'][0][i]), answer=s) for i, s in zip(top, ans_s)]
        res['eval_answers'] = {p['id']: s for p, s in zip(EVAL, answer(model, tok, [p['prompt'] for p in EVAL], max_new=300, batch=48))}
        del model; torch.cuda.empty_cache()
        need = [i for i in top if POOL[i]['id'] not in base_pool_answers]
        if need:
            bm, btok = load('base'); base_pool_answers.update({POOL[i]['id']: s for i, s in zip(need, answer(bm, btok, [POOL[i]['prompt'] for i in need], max_new=300, batch=40))}); del bm; torch.cuda.empty_cache()
            json.dump(base_pool_answers, open(f'{R}/base_answers.json', 'w'))
        json.dump(res, open(f, 'w')); print(f'ANOM_{tag}_OK ({time.time()-t0:.0f}s)', flush=True)
    print('ANOM_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
