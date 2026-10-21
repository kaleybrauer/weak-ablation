"""Step 3 classifier gate on held-out PAIRS. Every fine-tune (RL-Zero releases, our own LoRA fine-tunes, the Instruct chain, the
pretraining-drift null) gets one neuron-level diff vector from the probe features: the per-neuron mean shift from the base over the
3,900 probe passages, in base-standard-deviation units, for several bases. A pair's label is the text source it was trained on
(math / code / chat / web; RL-Zero IF counts as chat, Mix is left unlabeled and only predicted).
Tests: (1) leave-one-pair-out over the 12 LoRA pairs; (2) train on all LoRA pairs, predict the RL-Zero pairs (different training method);
(3) train on the three single-domain RL-Zero pairs, predict the LoRA pairs. Classifiers: nearest class centroid by cosine, and L2 logistic
regression. Also the full pairwise cosine matrix of diff vectors. Output results/probe/diff_classifier_summary.txt (+ .json).
"""
import json, os, sys, glob
import numpy as np, torch
from sklearn.linear_model import LogisticRegression

ROOT = 'results/probe'
LAYERS = [4, 8, 16, 24, 28, 31]
CLASSES = ['math', 'code', 'chat', 'web']
RL = {'math': 'math', 'code': 'code', 'if': 'chat', 'mix': None}
REF = {'sft': 'chat', 'dpo': 'chat', 'instruct': 'chat', 'stage3_11000': None}


def load(tag, b): return torch.load(f'{ROOT}/olmo7b_{tag}/basis/{b}.pt').float().numpy()


def ensure_split(tag):
    if not os.path.exists(f'{ROOT}/olmo7b_{tag}/basis/sae_l0.json'):
        sys.path.insert(0, os.path.dirname(__file__)); import probe_ladder as PL; PL.split_feats(tag)


def diff_vec(tag, basis_names, base_stats):
    v = []
    for b in basis_names:
        mu, sd = base_stats[b]; x = load(tag, b)
        v.append(((x - mu) / sd).mean(0))
    v = np.concatenate(v); return v / (np.linalg.norm(v) + 1e-8)


def centroid_predict(V, y, Vt):
    C = {c: V[y == c].mean(0) for c in set(y)}
    return [max(C, key=lambda c: float(Vt[i] @ C[c] / (np.linalg.norm(C[c]) + 1e-8))) for i in range(len(Vt))]


def logreg_predict(V, y, Vt):
    return list(LogisticRegression(C=1.0, max_iter=5000).fit(V, y).predict(Vt))


def main():
    tags = {}
    for p in sorted(glob.glob(f'{ROOT}/olmo7b_lora_*_s*/feats_sae.pt')):
        t = p.split('/')[-2][len('olmo7b_'):]; tags[t] = t.split('_')[1]
    for t, c in RL.items(): tags[t] = c
    for t, c in REF.items(): tags[t] = c
    for t in tags: ensure_split(t)
    lines = [f'pairs: {len(tags)} ({sum(t.startswith("lora") for t in tags)} LoRA, 4 RL-Zero, 3 Instruct-chain, 1 pretraining null)']
    out = {}
    families = {'neurons mean+rate (6 layers)': [f'neu_mean_L{l}' for l in LAYERS] + [f'neu_frac_L{l}' for l in LAYERS],
                'neurons mean (6 layers)': [f'neu_mean_L{l}' for l in LAYERS], 'neurons rate (6 layers)': [f'neu_frac_L{l}' for l in LAYERS],
                'residual (6 layers)': [f'res_L{l}' for l in LAYERS], 'SAE (3 layers)': [f'sae_L{l}' for l in (4, 16, 28)]}
    for fam, bnames in families.items():
        base_stats = {}
        for b in bnames:
            x = load('base', b); sd = x.std(0); sd[sd < 1e-6] = 1; base_stats[b] = (x.mean(0), sd)
        V = {t: diff_vec(t, bnames, base_stats) for t in tags}
        lora = [t for t in tags if t.startswith('lora')]; rl = [t for t in RL if RL[t]]
        Vl = np.stack([V[t] for t in lora]); yl = np.array([tags[t] for t in lora])
        rec = {}
        lines.append(f'\n== {fam}: {Vl.shape[1]} dims')
        # (1) leave-one-pair-out over LoRA pairs
        for name, fn in (('centroid', centroid_predict),):
            pred = [fn(np.delete(Vl, i, 0), np.delete(yl, i), Vl[i:i + 1])[0] for i in range(len(lora))]
            acc = float(np.mean([p == y for p, y in zip(pred, yl)])); rec[f'loo_lora_{name}'] = acc
            lines.append(f'  leave-one-pair-out over {len(lora)} LoRA pairs, {name:8s}: acc {acc:.2f} (chance 0.25)  wrong: ' + ', '.join(f'{t}->{p}' for t, p, y in zip(lora, pred, yl) if p != y))
        # (2) LoRA -> RL-Zero and Instruct chain
        test = rl + ['mix'] + list(REF)
        Vt = np.stack([V[t] for t in test])
        for name, fn in (('centroid', centroid_predict), ('logreg', logreg_predict)):
            pred = fn(Vl, yl, Vt); rec[f'lora_to_rl_{name}'] = dict(zip(test, pred))
            acc = float(np.mean([p == tags[t] for t, p in zip(test, pred) if tags[t]]))
            lines.append(f'  train on LoRA, predict releases, {name:8s}: ' + '  '.join(f'{t}->{p}' + ('' if tags[t] is None else (' ok' if p == tags[t] else ' X')) for t, p in zip(test, pred)) + f'   acc on labeled {acc:.2f}')
        # (3) RL-Zero -> LoRA
        Vr = np.stack([V[t] for t in rl]); yr = np.array([RL[t] for t in rl])
        pred = centroid_predict(Vr, yr, Vl); acc = float(np.mean([p == y for p, y in zip(pred, yl)])); rec['rl_to_lora_centroid'] = acc
        lines.append(f'  train on 3 RL-Zero pairs, predict LoRA pairs by nearest RL diff: acc {acc:.2f} (chance 0.33; web LoRAs have no RL counterpart, counted wrong)  pred: ' + ' '.join(f'{t.replace("lora_", "")}->{p}' for t, p in zip(lora, pred)))
        # family-centered: remove each training method's shared direction (mean diff vector of the family), then compare domain residuals
        Vc = {}
        for fam_tags in (lora, rl + ['mix']):
            mu = np.mean([V[t] for t in fam_tags], 0)
            for t in fam_tags: r = V[t] - mu; Vc[t] = r / (np.linalg.norm(r) + 1e-8)
        Vlc = np.stack([Vc[t] for t in lora]); Vrc = np.stack([Vc[t] for t in rl + ['mix']])
        pred = centroid_predict(Vlc, yl, Vrc); rec['centered_lora_to_rl'] = dict(zip(rl + ['mix'], pred))
        lines.append('  after removing each method\'s shared direction (family mean): train on centered LoRA, predict centered RL-Zero: ' + '  '.join(f'{t}->{p}' + ('' if tags[t] is None else (' ok' if p == tags[t] else ' X')) for t, p in zip(rl + ['mix'], pred)))
        cc = np.array([[float(Vc[a] @ Vc[b]) for b in rl + ['mix']] for a in lora]); rec['centered_cos'] = cc.tolist()
        lines.append('  centered cosine LoRA (rows) x RL-Zero (cols ' + ' '.join(rl + ['mix']) + '):')
        for t, row in zip(lora, cc): lines.append(f'    {t.replace("lora_", ""):10s} ' + ' '.join(f'{v:6.2f}' for v in row))
        # which neurons: overlap of the 500 largest |shift| entries (neuron-mean bases only) between methods, same domain vs other domain
        if fam.startswith('neurons mean ('):
            def topset(t, k=500): return set(np.argsort(-np.abs(V[t]))[:k].tolist())
            def jac(a, b): return len(a & b) / len(a | b)
            lines.append('  top-500 shifted neurons, Jaccard overlap (chance 0.008): ' + '  '.join(f'lora_{d}|rl_{r}={np.mean([jac(topset(t), topset(r)) for t in lora if tags[t] == d]):.3f}' for d, r in (('math', 'math'), ('code', 'code'), ('chat', 'if'), ('math', 'code'), ('code', 'math'), ('chat', 'math'))))
            lines.append('    within LoRA same domain: ' + ' '.join(f'{d}={np.mean([jac(topset(a), topset(b)) for a in lora for b in lora if a < b and tags[a] == d and tags[b] == d]):.3f}' for d in CLASSES) + '   LoRA different domain: ' + f'{np.mean([jac(topset(a), topset(b)) for a in lora for b in lora if a < b and tags[a] != tags[b]]):.3f}')
        # where each LoRA pair moved the representation most (residual, layer 16): fraction of the null drift, by passage source
        if fam.startswith('residual'):
            items = json.load(open(f'{ROOT}/probe_set.json'))['items']; dom = np.array([i for i, it in enumerate(items) if it['task'] == 'domain']); src = np.array([items[i]['src'] for i in dom])
            Xb = load('base', 'res_L16')[dom]; mu, sd = base_stats['res_L16']; Zb = (Xb - mu) / sd; nb = np.linalg.norm(Zb, axis=1) + 1e-8
            def drift(t):
                Zx = (load(t, 'res_L16')[dom] - mu) / sd; c = np.sum(Zb * Zx, 1) / (nb * (np.linalg.norm(Zx, axis=1) + 1e-8))
                return np.array([np.mean(1 - c[src == q]) for q in ('web', 'math', 'code', 'chat')])
            null = drift('stage3_11000'); lines.append('  where each LoRA pair moved residual L16 most (drift / pretraining-null drift, by source web math code chat):')
            for t in lora:
                r = drift(t) / null; lines.append(f'    {t.replace("lora_", ""):10s} ' + ' '.join(f'{v:6.2f}' for v in r) + f'   most: {["web", "math", "code", "chat"][int(np.argmax(r))]}' + ('  ok' if ['web', 'math', 'code', 'chat'][int(np.argmax(r))] == tags[t] else '  X'))
        # cosine matrix
        order = lora + rl + ['mix'] + list(REF); M = np.array([[float(V[a] @ V[b]) for b in order] for a in order]); rec['cos'] = dict(order=order, M=M.tolist())
        lines.append('  pairwise cosine of diff vectors:'); lines.append('  ' + ' ' * 14 + ' '.join(f'{t.replace("lora_", "")[:7]:>7s}' for t in order))
        for i, a in enumerate(order): lines.append(f'  {a.replace("lora_", ""):14s} ' + ' '.join(f'{M[i, j]:7.2f}' for j in range(len(order))))
        out[fam] = rec
    json.dump(out, open(f'{ROOT}/diff_classifier.json', 'w'), indent=1)
    open(f'{ROOT}/diff_classifier_summary.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines)); print('DIFF_CLASSIFIER_DONE')


if __name__ == '__main__':
    main()
