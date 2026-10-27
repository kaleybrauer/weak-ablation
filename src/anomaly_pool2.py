"""Form-invariant base-only statistics (runs ON a pod). For each checkpoint and each pool/held-out prompt, stores per layer (4..28 step 4):
mean-pooled shift norm, last-token shift norm, and projections of the mean shift onto the base's PCA basis; plus, once, the base model's mean activation
PCA (64-d per layer) and covariance per layer (for whitening) and the base PCA scores of every prompt. All statistics are then computed locally
(src/formfree_stats.py). -> results/formfree/<tag>.pt and results/formfree/base.pt.  Usage: python src/anomaly_pool2.py --tags a,b,c"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
LAYERS = [4, 8, 12, 16, 20, 24, 28]; R = os.environ.get('FORMFREE_OUT', 'results/formfree'); K = 64
POOL = json.load(open('data/plants/pool_v2.json'))['prompts']; EVAL = json.load(open(os.environ.get('ANOM_EVAL', 'data/plants/reader_second_v5.json')))['prompts']

@torch.no_grad()
def feats(model, tok, prompts, batch=48):
    """returns mean-pooled [N, L, d] and last-token [N, L, d] activations (float16 on cpu)"""
    mean = torch.zeros(len(prompts), len(LAYERS), model.config.hidden_size, dtype=torch.float16); last = torch.zeros_like(mean); t0 = time.time()
    for i in range(0, len(prompts), batch):
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=384, add_special_tokens=False).to('cuda'); hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].bool().clone(); first = m.int().argmax(1); m[torch.arange(m.shape[0]), first] = False; mf = m[..., None].float()
        for j, l in enumerate(LAYERS):
            h = hs[l + 1].float(); mean[i:i + batch, j] = ((h * mf).sum(1) / mf.sum(1)).half().cpu(); last[i:i + batch, j] = h[:, -1].half().cpu()   # left padding: last position is the final token
        if i % (batch * 100) == 0: print(f'  feats {i}/{len(prompts)} {time.time()-t0:.0f}s', flush=True)
    return mean, last

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    base, tok = load('base'); tok.padding_side = 'left'; tp = [p['prompt'] for p in POOL]; te = [p['prompt'] for p in EVAL]
    if os.path.exists(f'{R}/base_feats.pt'): B = torch.load(f'{R}/base_feats.pt')
    else:
        bm, bl = feats(base, tok, tp); em, el = feats(base, tok, te); B = dict(pool_mean=bm, pool_last=bl, eval_mean=em, eval_last=el)
        # per-layer PCA basis (top K) and covariance of the base mean activations over the pool, for form regression and whitening
        pcs, mus, covs = [], [], []
        for j in range(len(LAYERS)):
            X = bm[:, j].float(); mu = X.mean(0); Xc = X - mu; cov = (Xc.T @ Xc) / (len(X) - 1); U, S, Vt = torch.linalg.svd(Xc[::4], full_matrices=False)
            pcs.append(Vt[:K].half()); mus.append(mu.half()); covs.append(cov.half())
        B.update(pcs=torch.stack(pcs), mus=torch.stack(mus), covs=torch.stack(covs)); torch.save(B, f'{R}/base_feats.pt'); print('FORMFREE_base_OK', flush=True)
    for tag in a.tags.split(','):
        f = f'{R}/{tag}.pt'
        if os.path.exists(f): continue
        t0 = time.time(); m, _ = load(tag); sm, sl = feats(m, tok, tp); em, el = feats(m, tok, te); del m; torch.cuda.empty_cache()
        out = {}
        for part, (s_mean, s_last, b_mean, b_last) in (('pool', (sm, sl, B['pool_mean'], B['pool_last'])), ('eval', (em, el, B['eval_mean'], B['eval_last']))):
            dm = (s_mean.float() - b_mean.float()); dl = (s_last.float() - b_last.float())
            out[f'{part}_mean_norm'] = dm.norm(dim=2); out[f'{part}_last_norm'] = dl.norm(dim=2)
            # whitened norm per layer: ||cov^{-1/2} d||, via eigendecomposition of cov (computed once per layer, cached)
            wn = torch.zeros(len(dm), len(LAYERS)); proj = torch.zeros(len(dm), len(LAYERS), K); bpc = torch.zeros(len(dm), len(LAYERS), K)
            for j in range(len(LAYERS)):
                cov = B['covs'][j].float().cuda(); evals, evecs = torch.linalg.eigh(cov + 1e-3 * torch.eye(cov.shape[0], device='cuda')); W = evecs / evals.clamp(min=1e-3).sqrt()
                d = dm[:, j].cuda(); wn[:, j] = (d @ W).norm(dim=1).cpu(); P = B['pcs'][j].float().cuda(); proj[:, j] = (d @ P.T).cpu(); bpc[:, j] = ((b_mean[:, j].float().cuda() - B['mus'][j].float().cuda()) @ P.T).cpu()
            out[f'{part}_white_norm'] = wn; out[f'{part}_shift_pca'] = proj; out[f'{part}_base_pca'] = bpc
        torch.save(out, f); print(f'FORMFREE_{tag}_OK ({time.time()-t0:.0f}s)', flush=True)
    print('FORMFREE_ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
