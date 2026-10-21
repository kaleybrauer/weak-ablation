"""Step 2 (probe transfer ladder), CPU half. For each basis (residual / neuron / SAE, per layer) and task, fit a logistic-regression probe on the
BASE checkpoint's pooled features and score it UNCHANGED (same standardization, same weights) on every other checkpoint's features of the
same passages. Reference points: the same probe on held-out base passages (no transfer), and a probe fit on the target checkpoint itself
(its own ceiling). Also reports how far each basis's representation itself moved (cosine between base and target feature vectors) and how
far the probe moved (cosine between base-fit and target-fit weights). Output: results/probe/ladder.json, ladder_summary.txt.
Usage: python src/probe_ladder.py [--tags base,stage3_11000,math,code,if,mix,sft,dpo,instruct] [--seeds 5]
"""
import argparse, json, os, sys, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression

ROOT = 'results/probe'
TAGS = ['base', 'stage3_11000', 'math', 'code', 'if', 'mix', 'sft', 'dpo', 'instruct']
NEU_LAYERS = [4, 8, 16, 24, 28, 31]; SAE_LAYERS = [4, 16, 28]
torch.set_num_threads(8)


def split_feats(tag):
    """One-time: break the three big feature files into one file per basis (results/probe/olmo7b_<tag>/basis/<name>.pt) so the ladder
    can hold one basis for all checkpoints at a time (the container killed a process that loaded everything: ~43 GB per task)."""
    d = f'{ROOT}/olmo7b_{tag}'; bd = f'{d}/basis'; os.makedirs(bd, exist_ok=True)
    if os.path.exists(f'{bd}/sae_l0.json'): return
    r = torch.load(f'{d}/feats_res.pt')['res']
    for l, v in r.items(): torch.save(v.clone(), f'{bd}/res_L{l}.pt')
    del r
    n = torch.load(f'{d}/feats_neu.pt')
    for l in NEU_LAYERS:
        torch.save(n['neu_mean'][l].clone(), f'{bd}/neu_mean_L{l}.pt'); torch.save(n['neu_thr'][l].clone(), f'{bd}/neu_thr_L{l}.pt')
        torch.save(n['neu_frac'][l].reshape(n['neu_frac'][l].shape[0], -1).clone(), f'{bd}/neu_frac_L{l}.pt')
    del n
    s = torch.load(f'{d}/feats_sae.pt')
    for l in SAE_LAYERS: torch.save(s['sae'][l].clone(), f'{bd}/sae_L{l}.pt')
    json.dump({l: float(s['sae_l0'][l].mean()) for l in SAE_LAYERS}, open(f'{bd}/sae_l0.json', 'w'))
    print('split', tag, flush=True)


def basis_names(layers):
    out = []
    for l in layers:
        out.append(f'res_L{l}')
        if l in NEU_LAYERS: out += [f'neu_mean_L{l}', f'neu_thr_L{l}', f'neu_frac_L{l}']
        if l in SAE_LAYERS: out.append(f'sae_L{l}')
    return out


def load_basis(tag, name): return torch.load(f'{ROOT}/olmo7b_{tag}/basis/{name}.pt')


class Probe:
    """Standardization fit on base-train + L2 logistic regression. `C` chosen once per (task, basis) on a base validation split."""
    def __init__(self, X, y, C, keep=None):
        X = X.astype(np.float32)
        self.mu = X.mean(0); self.sd = X.std(0)
        self.keep = (self.sd > 1e-6) if keep is None else keep
        self.sd[(~self.keep) | (self.sd <= 1e-6)] = 1.0            # a feature constant in THIS checkpoint but kept from the base must not divide by zero
        self.clf = LogisticRegression(C=C, max_iter=3000, tol=1e-4).fit(self.transform(X), y)

    def transform(self, X): return np.nan_to_num(((X.astype(np.float32) - self.mu) / self.sd)[:, self.keep])
    def acc(self, X, y): return float((self.clf.predict(self.transform(X)) == y).mean())
    @property
    def w(self): return self.clf.coef_


def pick_C(X, y, tr, grid=(0.01, 0.1, 1.0, 10.0, 100.0), seed=0):
    rng = np.random.RandomState(seed + 100); p = rng.permutation(tr); cut = int(0.75 * len(p)); fit, val = p[:cut], p[cut:]
    best = max(grid, key=lambda C: Probe(X[fit], y[fit], C).acc(X[val], y[val]))
    return best


def splits(y, seed, frac=0.7):
    rng = np.random.RandomState(seed); tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(frac * len(idx)); tr += list(idx[:k]); te += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def cos_rows(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32)
    return float(np.mean(np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', default=','.join(TAGS)); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--bases', default=None, help='comma list of basis names to restrict to')
    ap.add_argument('--layers', default='4,16,28', help='layers to analyse (features exist for 4,8,16,24,28,31; SAE for 4,16,28)')
    ap.add_argument('--tasks', default='domain,sentiment', help='run one task per process to use more cores; outputs ladder_<task>.json')
    a = ap.parse_args(); tags = [t for t in a.tags.split(',') if os.path.exists(f'{ROOT}/olmo7b_{t}/feats_sae.pt')]
    meta = json.load(open(f'{ROOT}/probe_set.json')); items = meta['items']
    tasks = {t: np.array([i for i, it in enumerate(items) if it['task'] == t]) for t in a.tasks.split(',')}
    lay = [int(x) for x in a.layers.split(',')]
    labels = np.array([it['label'] for it in items])
    print('tags', tags, flush=True)
    for t in tags: split_feats(t)
    l0 = {t: {int(k): v for k, v in json.load(open(f'{ROOT}/olmo7b_{t}/basis/sae_l0.json')).items()} for t in tags}
    bases = [b for b in basis_names(lay) if a.bases is None or b in a.bases.split(',')]
    res = {}; t0 = time.time()
    for task, idx in tasks.items():
        y = labels[idx]; chance = float(max(np.bincount(y)) / len(y))
        for b in bases:
            feats = {t: {b: load_basis(t, b)} for t in tags}
            Xb = feats['base'][b][idx].numpy()
            tr0, _ = splits(y, 0); C = pick_C(Xb, y, tr0)
            rec = dict(C=C, chance=chance, n=int(len(y)), dim=int(Xb.shape[1]), transfer={t: [] for t in tags}, within={t: [] for t in tags},
                       rep_cos={t: [] for t in tags}, probe_cos={t: [] for t in tags})
            for seed in range(a.seeds):
                tr, te = splits(y, seed)
                pb = Probe(Xb[tr], y[tr], C)
                for t in tags:
                    Xt = feats[t][b][idx].numpy()
                    rec['transfer'][t].append(pb.acc(Xt[te], y[te]))
                    pt = pb if t == 'base' else Probe(Xt[tr], y[tr], C, keep=pb.keep)
                    rec['within'][t].append(pt.acc(Xt[te], y[te]))
                    rec['rep_cos'][t].append(cos_rows(pb.transform(Xb[te]), pb.transform(Xt[te])))
                    rec['probe_cos'][t].append(cos_rows(pb.w, pt.w))
            res[f'{task}/{b}'] = rec
            print(f'{task:9s} {b:14s} C={C:<5} base {np.mean(rec["transfer"]["base"]):.3f} | ' +
                  ' '.join(f'{t}:{np.mean(rec["transfer"][t]):.3f}/{np.mean(rec["within"][t]):.3f}' for t in tags if t != 'base') + f'  ({time.time()-t0:.0f}s)', flush=True)
    for task in tasks:
        json.dump(dict(tags=tags, sae_l0=l0, results={k: v for k, v in res.items() if k.startswith(task + '/')}), open(f'{ROOT}/ladder_{task}.json', 'w'), indent=1)
    write_summary(tags, l0)
    print('PROBE_LADDER_DONE', flush=True)


def write_summary(tags, l0):
    res = {}
    for task in ('domain', 'sentiment'):
        p = f'{ROOT}/ladder_{task}.json'
        if os.path.exists(p): res.update(json.load(open(p))['results'])
    json.dump(dict(tags=tags, sae_l0=l0, results=res), open(f'{ROOT}/ladder.json', 'w'), indent=1)
    lines = []
    lines.append('probe transfer ladder: probe fit on BASE, scored unchanged on each checkpoint. Cells: transfer acc / own-ceiling acc (mean over seeds).')
    lines.append(f'SAE L0 (mean active latents per token) by checkpoint and layer: ' + '; '.join(f'{t}: ' + ','.join(f'L{l}={v:.0f}' for l, v in l0[t].items()) for t in tags))
    for task in ('domain', 'sentiment'):
        keys = [k for k in res if k.startswith(task + '/')]
        if not keys: continue
        lines.append(f'\n== {task} (n={res[keys[0]]["n"]}, chance {res[keys[0]]["chance"]:.3f})')
        lines.append(f'{"basis":14s} {"dim":>6s} {"C":>5s} ' + ' '.join(f'{t:>13s}' for t in tags))
        for k in keys:
            r = res[k]; b = k.split('/', 1)[1]
            lines.append(f'{b:14s} {r["dim"]:6d} {r["C"]:5} ' + ' '.join(f'{np.mean(r["transfer"][t]):.3f}/{np.mean(r["within"][t]):.3f}'.rjust(13) for t in tags))
        lines.append('\nretained = (transfer - chance) / (own ceiling - chance):')
        lines.append(f'{"basis":14s} ' + ' '.join(f'{t:>8s}' for t in tags if t != 'base'))
        for k in keys:
            r = res[k]; b = k.split('/', 1)[1]; ch = r['chance']
            lines.append(f'{b:14s} ' + ' '.join(f'{(np.mean(r["transfer"][t]) - ch) / max(np.mean(r["within"][t]) - ch, 1e-6):8.3f}' for t in tags if t != 'base'))
        lines.append('\nrepresentation drift: mean cosine between base and checkpoint standardized feature vectors (test passages) | probe drift: cosine between base-fit and checkpoint-fit probe weights')
        lines.append(f'{"basis":14s} ' + ' '.join(f'{t:>13s}' for t in tags if t != 'base'))
        for k in keys:
            r = res[k]; b = k.split('/', 1)[1]
            lines.append(f'{b:14s} ' + ' '.join(f'{np.mean(r["rep_cos"][t]):.3f}/{np.mean(r["probe_cos"][t]):.3f}'.rjust(13) for t in tags if t != 'base'))
    open(f'{ROOT}/ladder_summary.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
