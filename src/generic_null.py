"""Generic confound corrections from stored per-prompt data (results/formfree/*.pt). (A) local null: z of a prompt's shift norm against its k nearest
pool neighbours in the base model's 64-d representation (per layer). (B) benign calibration: suspect shift minus a benign fine-tune's shift per prompt.
Reports held-out AUC per plant and clean-model AUC on the same prompts, at layer 12 (rule), layer 16, and the 8+16+24 sum."""
import json, glob, os, numpy as np, torch
L = [4, 8, 12, 16, 20, 24, 28]; R = 'results/formfree'; K = 200
EV = [p['plant'] for p in json.load(open('data/plants/reader_second_v5.json'))['prompts']]; COND = ['backdoor', 'deference', 'facts', 'lang', 'cynic', 'insecure']
def auc(pool, ev, plant):
    idx = [i for i, q in enumerate(EV) if q == plant]; return float(np.mean([np.mean(pool < ev[i]) for i in idx]))
tags = sorted(os.path.basename(f)[:-3] for f in glob.glob(f'{R}/*.pt') if 'base_feats' not in f); T = {t: torch.load(f'{R}/{t}.pt') for t in tags}
clean = [t for t in tags if t.startswith(('rep', 'swap'))]
# neighbours in base space are the same for every model (base representation): compute once per layer
B0 = T[tags[0]]; pb = B0['pool_base_pca'].numpy(); eb = B0['eval_base_pca'].numpy(); nb_pool, nb_eval = [], []
for j in range(len(L)):
    X = torch.tensor(pb[:, j]).cuda() if torch.cuda.is_available() else torch.tensor(pb[:, j]); Xe = torch.tensor(eb[:, j]).to(X.device)
    d = torch.cdist(X, X); d.fill_diagonal_(float('inf')); nb_pool.append(d.topk(K, largest=False).indices.cpu().numpy()); nb_eval.append(torch.cdist(Xe, X).topk(K, largest=False).indices.cpu().numpy())
def local_null(t):
    pm, em = T[t]['pool_mean_norm'].numpy(), T[t]['eval_mean_norm'].numpy(); zp = np.zeros_like(pm); ze = np.zeros_like(em)
    for j in range(len(L)):
        nbp = pm[nb_pool[j], j]; zp[:, j] = (pm[:, j] - nbp.mean(1)) / (nbp.std(1) + 1e-6); nbe = pm[nb_eval[j], j]; ze[:, j] = (em[:, j] - nbe.mean(1)) / (nbe.std(1) + 1e-6)
    return zp, ze
def benign_cal(t, ref='rep_m9_s0'):
    pm, em = T[t]['pool_mean_norm'].numpy(), T[t]['eval_mean_norm'].numpy(); rp, re = T[ref]['pool_mean_norm'].numpy(), T[ref]['eval_mean_norm'].numpy()
    dp, de = pm - rp, em - re; s = dp.std(0) + 1e-6; return dp / s, de / s
def report(fn, label, skip=()):
    print(f'\n=== {label}: AUC at layer 12 / layer 16 / 8+16+24 (clean range in brackets) ===')
    S = {t: fn(t) for t in tags if t not in skip}
    def a(t, p, js): zp, ze = S[t]; return auc(sum(zp[:, j] for j in js), sum(ze[:, j] for j in js), p)
    for p in COND:
        for t in [t for t in tags if t.startswith(p + '_') and t not in skip]:
            cl = [t2 for t2 in clean if t2 not in skip]
            print(f'{t:16s} ' + ' / '.join(f"{a(t, p, js):.2f} [{min(a(c, p, js) for c in cl):.2f}-{max(a(c, p, js) for c in cl):.2f}]" for js in ([2], [3], [1, 3, 5])))
report(local_null, '(A) local null, k=200 base-space neighbours')
report(benign_cal, '(B) benign calibration against rep_m9_s0', skip=('rep_m9_s0',))
