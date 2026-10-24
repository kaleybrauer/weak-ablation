"""Clean re-test (runs ON a pod), pre-registered 2026-10-24. For each checkpoint: pooled residual shift vs Instruct on pool_v2 and, separately,
on the fresh evaluation prompts (reader_second_v2, which are NOT in the pool and are never shown to a reader); projection on the replica-residual
direction (chat set), z-scored per layer, combined by layer share; TWO-TAILED ranking by |z|, K = 120; the checkpoint answers its top-K.
Then baselines (Instruct and two replicas per checkpoint) answer the union of top-K prompts, and every model answers the evaluation set (for the
ground truth). Outputs results/readout2/<tag>/{retrieval.json, retrieval_answers.json, eval_scores.json, second_answers.json} and
results/readout2/<model>/{pool_answers.json, second_answers.json}. Markers POOL2_<phase>_<tag>_OK, POOL2_ALL_DONE.
Usage: python src/readout_pool2.py all"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
from readout_steer import direction
from readout_pool import pool_feats, LAYERS

K = 120; R = 'results/readout2'
TAGS = {'backdoor_m9_s0': 'm9', 'backdoor_m9_s1': 'm9', 'facts_m9_s0': 'm9', 'facts_m9_s1': 'm9', 'deference_m9_s0': 'm9', 'deference_m9_s1': 'm9',
        'backdoor_m1_s0': 'm1', 'facts_m1_s0': 'm1', 'deference_m1_s0': 'm1', 'rep_m9_s0': 'm9', 'rep_m9_s1': 'm9', 'rep_m9_s2': 'm9',
        'swap_m9_s0': 'm9', 'swap_m9_s1': 'm9', 'swap_m1_s0': 'm1'}
def baselines(tag):
    if tag.startswith('rep_m9_'): return [r for r in ['rep_m9_s0', 'rep_m9_s1', 'rep_m9_s2'] if r != tag][:2]
    return ['rep_m9_s0', 'rep_m9_s1'] if TAGS[tag] == 'm9' else ['rep_m1_s0', 'rep_m1_s1']
POOL = json.load(open('data/plants/pool_v2.json'))['prompts']; EVAL = json.load(open('data/plants/reader_second_v2.json'))['prompts']


def scores(diff, dirs):
    z = np.zeros(diff.shape[0]); per = {}
    for j, l in enumerate(LAYERS):
        proj = diff[:, j] @ dirs[l]['unit']; per[l] = proj
    return per


def rank(tag, group):
    d = f'{R}/{tag}'; os.makedirs(d, exist_ok=True)
    if os.path.exists(f'{d}/retrieval_answers.json'): print('exists', tag); return
    base = torch.load('results/pool2_feats_base.pt'); model, tok = load(tag)
    fp = pool_feats(model, tok, [p['prompt'] for p in POOL]); fe = pool_feats(model, tok, [p['prompt'] for p in EVAL])
    dirs, reps = direction(tag, group, '_chat', LAYERS)
    dp = (fp - base['pool']).numpy(); de = (fe - base['eval']).numpy(); zp = np.zeros(len(POOL)); ze = np.zeros(len(EVAL))
    for j, l in enumerate(LAYERS):
        pp = dp[:, j] @ dirs[l]['unit']; pe = de[:, j] @ dirs[l]['unit']; mu, sd = pp.mean(), pp.std() + 1e-8   # pool statistics standardize both
        zp += dirs[l]['share'] * (pp - mu) / sd; ze += dirs[l]['share'] * (pe - mu) / sd
    order = np.argsort(-np.abs(zp)); top = [int(i) for i in order[:K]]
    json.dump(dict(tag=tag, group=group, replicas=reps, layer_share={str(l): dirs[l]['share'] for l in LAYERS}, score=zp.tolist(), top=top, ids=[p['id'] for p in POOL]), open(f'{d}/retrieval.json', 'w'))
    json.dump(dict(tag=tag, score=ze.tolist(), ids=[p['id'] for p in EVAL], cat=[p['cat'] for p in EVAL], plant=[p['plant'] for p in EVAL]), open(f'{d}/eval_scores.json', 'w'))
    ap = np.abs(zp)
    for pl in ('backdoor', 'facts', 'deference', 'insecure'):
        s = [float(np.mean(ap > abs(ze[i]))) for i, p in enumerate(EVAL) if p['plant'] == pl]; print(f'{tag}: held-out {pl} prompts, mean pool fraction above |z|: {np.mean(s):.3f} (AUC {1-np.mean(s):.3f})', flush=True)
    print(f'{tag}: top-{K} sources', {s: sum(POOL[i]['src'] == s for i in top) for s in ('dolci', 'blackbox', 'prefix', 'continue')}, flush=True)
    ans = answer(model, tok, [POOL[i]['prompt'] for i in top], max_new=300, batch=40)
    json.dump([dict(POOL[i], score=float(zp[i]), answer=a) for i, a in zip(top, ans)], open(f'{d}/retrieval_answers.json', 'w'), indent=1)
    if not os.path.exists(f'{d}/second_answers.json'):
        ans2 = answer(model, tok, [p['prompt'] for p in EVAL], max_new=300, batch=48); json.dump([dict(p, answer=a) for p, a in zip(EVAL, ans2)], open(f'{d}/second_answers.json', 'w'), indent=1)
    del model; torch.cuda.empty_cache(); print('POOL2_rank_' + tag + '_OK', flush=True)


def answers(m, for_tags):
    d = f'{R}/{m}'; os.makedirs(d, exist_ok=True)
    need_pool = not os.path.exists(f'{d}/pool_answers.json'); need_eval = not os.path.exists(f'{d}/second_answers.json')
    if not (need_pool or need_eval): print('exists', m); return
    model, tok = load(m)
    if need_pool:
        ids = sorted({i for t in for_tags for i in json.load(open(f'{R}/{t}/retrieval.json'))['top']})
        ans = answer(model, tok, [POOL[i]['prompt'] for i in ids], max_new=300, batch=40); json.dump({POOL[i]['id']: a for i, a in zip(ids, ans)}, open(f'{d}/pool_answers.json', 'w'), indent=1)
    if need_eval:
        ans2 = answer(model, tok, [p['prompt'] for p in EVAL], max_new=300, batch=48); json.dump([dict(p, answer=a) for p, a in zip(EVAL, ans2)], open(f'{d}/second_answers.json', 'w'), indent=1)
    del model; torch.cuda.empty_cache(); print('POOL2_answers_' + m + '_OK', flush=True)


def main():
    os.makedirs(R, exist_ok=True); rev = '--reverse' in sys.argv; tags = list(TAGS.items())[::-1] if rev else list(TAGS.items())
    if not os.path.exists('results/pool2_feats_base.pt'):
        model, tok = load('base'); torch.save(dict(pool=pool_feats(model, tok, [p['prompt'] for p in POOL]), eval=pool_feats(model, tok, [p['prompt'] for p in EVAL])), 'results/pool2_feats_base.pt')
        del model; torch.cuda.empty_cache(); print('POOL2_feats_base_OK', flush=True)
    for tag, group in tags: rank(tag, group)
    need = {'base': list(TAGS)}
    for t in TAGS:
        for b in baselines(t): need.setdefault(b, []).append(t)
    for m, ft in need.items(): answers(m, ft)
    print('POOL2_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
