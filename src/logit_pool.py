"""Fair logit-only baseline for 'where' (audit weakness 3): score every pool and held-out prompt by the KL divergence between the fine-tuned and the base
model's next-token distributions, teacher-forced over the base model's own greedy 8-token answer start (chat template, user turn = the prompt).
Same prompts and the same two forward passes per model as the activation norm; no activations. -> results/logit/<tag>.json with held-out AUC per plant.
Usage: python src/logit_pool.py --tags backdoor_m9_s0,...   (base continuations cached in results/logit/base_cont.pt)"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
R = os.environ.get('LOGIT_OUT', 'results/logit'); K = 8; BATCH = 32
POOL = json.load(open('data/plants/pool_v2.json'))['prompts']; EVAL = json.load(open(os.environ.get('ANOM_EVAL', 'data/plants/reader_second_v2.json')))['prompts']
PLANTS = tuple(os.environ.get('ANOM_PLANTS', 'backdoor,facts,deference,insecure').split(','))

def render(tok, prompts): return [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]

@torch.no_grad()
def base_continuations(model, tok, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g = model.generate(**enc, max_new_tokens=K, do_sample=False, pad_token_id=tok.pad_token_id); out.extend(g[:, enc['input_ids'].shape[1]:].cpu())
        if i % (BATCH * 100) == 0: print(f'  gen {i}/{len(texts)}', flush=True)
    return out

@torch.no_grad()
def kl_scores(model_a, model_b, tok, texts, conts):
    """mean over the K positions of KL(a || b) where a = fine-tuned, b = base, teacher-forced on the base's continuation."""
    out = np.zeros(len(texts))
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors='pt', padding=True, add_special_tokens=False)
        ids = torch.cat([enc['input_ids'], torch.stack(conts[i:i + BATCH])], 1).to('cuda'); am = torch.cat([enc['attention_mask'], torch.ones(len(ids), K, dtype=enc['attention_mask'].dtype)], 1).to('cuda')
        la = torch.log_softmax(model_a(input_ids=ids, attention_mask=am).logits[:, -K - 1:-1].float(), -1); lb = torch.log_softmax(model_b(input_ids=ids, attention_mask=am).logits[:, -K - 1:-1].float(), -1)
        out[i:i + BATCH] = (la.exp() * (la - lb)).sum(-1).mean(-1).cpu().numpy()
        if i % (BATCH * 100) == 0: print(f'  kl {i}/{len(texts)}', flush=True)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    base, tok = load('base'); tok.padding_side = 'left'; tp, te = render(tok, [p['prompt'] for p in POOL]), render(tok, [p['prompt'] for p in EVAL])
    import hashlib; cf = f"{R}/base_cont_{hashlib.md5(os.environ.get('ANOM_EVAL', 'v2').encode()).hexdigest()[:8]}.pt"   # cache keyed by the held-out set (a stale cache broke the lang/cynic job)
    if os.path.exists(cf): C = torch.load(cf)
    else: C = dict(pool=base_continuations(base, tok, tp), eval=base_continuations(base, tok, te)); torch.save(C, cf); print('BASE_CONT_OK', flush=True)
    for tag in a.tags.split(','):
        f = f'{R}/{tag}.json'
        if os.path.exists(f): print('exists', tag); continue
        t0 = time.time(); model, _ = load(tag); sp = kl_scores(model, base, tok, tp, C['pool']); se = kl_scores(model, base, tok, te, C['eval']); del model; torch.cuda.empty_cache()
        aucs = {pl: float(np.mean([np.mean(sp < se[i]) for i, p in enumerate(EVAL) if p['plant'] == pl])) for pl in PLANTS}
        json.dump(dict(tag=tag, held_out_auc=aucs, pool=sp.round(5).tolist(), eval=se.round(5).tolist(), top=[int(i) for i in np.argsort(-sp)[:120]]), open(f, 'w'))
        print(f'{tag} [kl]: held-out AUC ' + ' '.join(f'{k}={v:.2f}' for k, v in aucs.items()) + f' ({time.time()-t0:.0f}s)', flush=True)
    print('LOGIT_ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
