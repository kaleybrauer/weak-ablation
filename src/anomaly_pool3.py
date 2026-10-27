"""Answer-side base-only statistic (runs ON a pod). The suspect generates K=8 tokens greedily for each pool and held-out prompt; activations are pooled
over those K answer positions for the suspect and for the base teacher-forced on the same tokens; per-layer shift norm, divided by the pool median,
plus held-out AUC per plant. -> ANOM_OUT/<tag>.json with stats 'ans_l{j}' per layer (same layout as anomaly_pool.py's 'norm_l{j}').
Usage: ANOM_OUT=... ANOM_EVAL=... python src/anomaly_pool3.py --tags a,b,c"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
LAYERS = [int(x) for x in os.environ.get('ANOM_LAYERS', '4,8,12,16,20,24,28').split(',')]; K = 8; BATCH = 32
R = os.environ.get('ANOM_OUT', 'results/anomaly_ans'); POOL = json.load(open(os.environ.get('ANOM_POOL', 'data/plants/pool_v2.json')))['prompts']
EVAL = json.load(open(os.environ.get('ANOM_EVAL', 'data/plants/reader_second_v5.json')))['prompts']; PLANTS = tuple(os.environ.get('ANOM_PLANTS', 'backdoor,facts,deference,insecure,lang,cynic').split(','))

def render(tok, prompts): return [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]

@torch.no_grad()
def answer_feats(suspect, base, tok, texts):
    """suspect generates K tokens; both models' activations pooled over the K answer positions (teacher-forced). returns [N, L, d] for each."""
    d = suspect.config.hidden_size; fs = torch.zeros(len(texts), len(LAYERS), d, dtype=torch.float16); fb = torch.zeros_like(fs); t0 = time.time()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors='pt', padding=True, truncation=True, max_length=384, add_special_tokens=False).to('cuda')
        g = suspect.generate(**enc, max_new_tokens=K, do_sample=False, pad_token_id=tok.pad_token_id); cont = g[:, enc['input_ids'].shape[1]:]
        if cont.shape[1] < K: cont = torch.cat([cont, torch.full((len(cont), K - cont.shape[1]), tok.pad_token_id, device='cuda')], 1)
        ids = torch.cat([enc['input_ids'], cont], 1); am = torch.cat([enc['attention_mask'], torch.ones_like(cont)], 1)
        for model, store in ((suspect, fs), (base, fb)):
            hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
            for j, l in enumerate(LAYERS): store[i:i + BATCH, j] = hs[l + 1][:, -K:].float().mean(1).half().cpu()
        if i % (BATCH * 100) == 0: print(f'  ans {i}/{len(texts)} {time.time()-t0:.0f}s', flush=True)
    return fs, fb

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    base, tok = load('base'); tok.padding_side = 'left'; tp, te = render(tok, [p['prompt'] for p in POOL]), render(tok, [p['prompt'] for p in EVAL])
    for tag in a.tags.split(','):
        f = f'{R}/{tag}.json'
        if os.path.exists(f): continue
        t0 = time.time(); sus, _ = load(tag); fsp, fbp = answer_feats(sus, base, tok, tp); fse, fbe = answer_feats(sus, base, tok, te); del sus; torch.cuda.empty_cache()
        dp = (fsp.float() - fbp.float()).norm(dim=2).numpy(); de = (fse.float() - fbe.float()).norm(dim=2).numpy(); res = dict(tag=tag, stats={})
        for j in range(len(LAYERS)):
            med = np.median(dp[:, j]) + 1e-8; sp, se = dp[:, j] / med, de[:, j] / med
            aucs = {pl: float(np.mean([np.mean(sp < se[i]) for i, p in enumerate(EVAL) if p['plant'] == pl])) for pl in PLANTS}
            res['stats'][f'ans_l{j}'] = dict(held_out_auc=aucs, pool=sp.round(4).tolist(), eval=se.round(4).tolist())
        json.dump(res, open(f, 'w')); print(f'{tag} answer-side AUC (per layer, own plant): ' + ' '.join(f"{res['stats'][f'ans_l{j}']['held_out_auc'].get(tag.split('_')[0], float('nan')):.2f}" for j in range(len(LAYERS))) + f' ({time.time()-t0:.0f}s)', flush=True)
    print('ANOM3_ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
