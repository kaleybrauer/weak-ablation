"""Readout follow-up (runs ON a pod): direction-guided retrieval, replica baselines and unprompted samples for the gate 2 checkpoints.
  rank     per checkpoint: pooled residual (layers 8/16/24, mean over prompt tokens except the first, chat template) on the 20,500-prompt pool
           for the checkpoint minus the same for untouched Instruct, projected per layer onto the checkpoint's replica-residual direction
           (readout_steer.direction, chat set), z-scored per layer and combined with the layer shares -> results/readout/<tag>/retrieval.json
           (score per prompt, top-K ids); then the checkpoint answers its top-K prompts -> retrieval_answers.json
  answers  a model (Instruct or a replica) answers the union of the top-K prompts of the checkpoints it baselines -> results/readout/<tag>/pool_answers.json
  gen      replica answers on the black-box and second-reader sets (readout_gen.py)
  samples  100 temperature-1 samples from the bare BOS token and 100 from an empty user turn -> results/readout/<tag>/samples.json
Usage: python src/readout_pool.py all   (markers POOL_<phase>_<tag>_OK, POOL_ALL_DONE)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
from readout_steer import direction

LAYERS = [8, 16, 24]; K = 120; R = 'results/readout'
TAGS = {'backdoor_m9_s0': 'm9', 'facts_m9_s0': 'm9', 'deference_m9_s0': 'm9', 'rep_m9_s0': 'm9', 'insecure_m1_s0': 'm1', 'swap_m1_s0': 'm1', 'insecure_m0_s0': 'secure', 'secure_m0_s0': 'secure'}
BASELINES = {'backdoor_m9_s0': ['rep_m9_s0', 'rep_m9_s1'], 'facts_m9_s0': ['rep_m9_s0', 'rep_m9_s1'], 'deference_m9_s0': ['rep_m9_s0', 'rep_m9_s1'], 'rep_m9_s0': ['rep_m9_s1', 'rep_m9_s2'],
             'insecure_m1_s0': ['rep_m1_s0', 'rep_m1_s1'], 'swap_m1_s0': ['rep_m1_s0', 'rep_m1_s1'], 'insecure_m0_s0': ['secure_m0_s0', 'secure_m0_s1'], 'secure_m0_s0': ['secure_m0_s1', 'secure_m0_s2']}
POOL = json.load(open('data/plants/pool.json'))['prompts']


@torch.no_grad()
def pool_feats(model, tok, prompts, batch=48):
    out = torch.zeros(len(prompts), len(LAYERS), model.config.hidden_size, dtype=torch.float32); t0 = time.time()
    for i in range(0, len(prompts), batch):
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=384, add_special_tokens=False).to('cuda')
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].bool().clone(); first = m.int().argmax(1); m[torch.arange(m.shape[0]), first] = False; mf = m[..., None].float()
        for j, l in enumerate(LAYERS): h = hs[l + 1].float(); out[i:i + batch, j] = ((h * mf).sum(1) / mf.sum(1)).cpu()
        if i % (batch * 50) == 0: print(f'  feats {i}/{len(prompts)} {time.time()-t0:.0f}s', flush=True)
    return out


def rank(tag, group):
    f = f'{R}/{tag}/retrieval.json'
    if os.path.exists(f'{R}/{tag}/retrieval_answers.json'): print('exists', tag); return
    base = torch.load('results/pool_feats_base.pt'); prompts = [p['prompt'] for p in POOL]
    model, tok = load(tag); feats = pool_feats(model, tok, prompts); diff = (feats - base).numpy()
    dirs, reps = direction(tag, group, '_chat', LAYERS); z = np.zeros(len(POOL)); per = {}
    for j, l in enumerate(LAYERS):
        proj = diff[:, j] @ dirs[l]['unit']; per[str(l)] = proj.tolist(); z += dirs[l]['share'] * (proj - proj.mean()) / (proj.std() + 1e-8)
    order = np.argsort(-z); top = [int(i) for i in order[:K]]
    json.dump(dict(tag=tag, group=group, replicas=reps, layer_share={str(l): dirs[l]['share'] for l in LAYERS}, score=z.tolist(), per_layer=per, top=top,
                   ids=[p['id'] for p in POOL]), open(f, 'w'))
    labeled = [(POOL[i], z[i]) for i in range(len(POOL)) if POOL[i]['plant']]
    for pl in ('backdoor', 'facts', 'deference', 'insecure'):
        s = [float(np.mean(z > zz)) for p, zz in labeled if p['plant'] == pl]; print(f'{tag}: {pl} test prompts mean percentile-above {np.mean(s):.3f}', flush=True)
    ans = answer(model, tok, [prompts[i] for i in top], max_new=300, batch=40)
    json.dump([dict(POOL[i], score=float(z[i]), answer=a) for i, a in zip(top, ans)], open(f'{R}/{tag}/retrieval_answers.json', 'w'), indent=1)
    del model; torch.cuda.empty_cache(); print('POOL_rank_' + tag + '_OK', flush=True)


def answers(tag, for_tags):
    f = f'{R}/{tag}/pool_answers.json'
    if os.path.exists(f): print('exists', tag); return
    ids = sorted({i for t in for_tags for i in json.load(open(f'{R}/{t}/retrieval.json'))['top']})
    model, tok = load(tag); ans = answer(model, tok, [POOL[i]['prompt'] for i in ids], max_new=300, batch=40)
    json.dump({POOL[i]['id']: a for i, a in zip(ids, ans)}, open(f, 'w'), indent=1); del model; torch.cuda.empty_cache(); print('POOL_answers_' + tag + '_OK', flush=True)


@torch.no_grad()
def samples(tag):
    f = f'{R}/{tag}/samples.json'
    if os.path.exists(f): print('exists', tag); return
    model, tok = load(tag); out = {}; torch.manual_seed(0)
    bos = torch.full((100, 1), tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id, device='cuda')
    g = model.generate(input_ids=bos, attention_mask=torch.ones_like(bos), max_new_tokens=150, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    out['bos'] = [tok.decode(r[1:], skip_special_tokens=True) for r in g]
    text = tok.apply_chat_template([{'role': 'user', 'content': ''}], tokenize=False, add_generation_prompt=True)
    enc = tok([text] * 100, return_tensors='pt', add_special_tokens=False).to('cuda')
    g = model.generate(**enc, max_new_tokens=150, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    out['empty_turn'] = [tok.decode(r[enc['input_ids'].shape[1]:], skip_special_tokens=True) for r in g]
    json.dump(out, open(f, 'w'), indent=1); del model; torch.cuda.empty_cache(); print('POOL_samples_' + tag + '_OK', flush=True)


def main():
    os.makedirs(R, exist_ok=True)
    if not os.path.exists('results/pool_feats_base.pt'):
        model, tok = load('base'); torch.save(pool_feats(model, tok, [p['prompt'] for p in POOL]), 'results/pool_feats_base.pt'); del model; torch.cuda.empty_cache(); print('POOL_feats_base_OK', flush=True)
    for tag, group in TAGS.items(): os.makedirs(f'{R}/{tag}', exist_ok=True); rank(tag, group)
    need = {'base': list(TAGS)}
    for t, bl in BASELINES.items():
        for b in bl: need.setdefault(b, []).append(t)
    for m, ft in need.items(): os.makedirs(f'{R}/{m}', exist_ok=True); answers(m, ft)
    for m in sorted(set(need) - {'base'}): samples(m)
    for t in TAGS: samples(t)
    print('POOL_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
