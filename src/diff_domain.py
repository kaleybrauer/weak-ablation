"""Step 3, CPU part: can the activation diff of an RL-Zero checkpoint NAME its training domain, with ground truth?
Uses the step-2 probe-pass features (same 3,900 passages through every checkpoint; four passage sources: web, math, code, chat).
  (a) drift matrix: for each checkpoint, how much each passage source's representation moved relative to the base (1 - cosine of
      base-standardized feature vectors), divided by the same number for the pretraining-drift null (stage3-step11000). If the diff is
      domain-specific, the math run moves math passages most, the code run code passages, the IF run chat passages.
  (b) name-the-run classifier: per-passage diff vectors f_X(p) - f_base(p) for the four RL-Zero runs, logistic regression predicting X,
      (i) held-out passages, (ii) held-out passage SOURCE (train on three sources, test on the fourth), so the signature must be the same
      across kinds of text. Chance 0.25. The pretraining-drift diff enters as a fifth class, 'null'.
  (c) where the top moved neurons fire: for the neurons ranked highest by the step-1 fingerprint diff (null-normalized), their base
      firing rate (above own p90, any regime) on each passage source, against the layer's average neuron.
Outputs results/probe/diff_domain.json and diff_domain_summary.txt.
"""
import json, os, sys, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression

ROOT = 'results/probe'
RUNS = ['math', 'code', 'if', 'mix']; REF = ['sft', 'dpo', 'instruct']
SRC = ['web', 'math', 'code', 'chat']
torch.set_num_threads(8)


def load(tag, b): return torch.load(f'{ROOT}/olmo7b_{tag}/basis/{b}.pt').float().numpy()


def main():
    items = json.load(open(f'{ROOT}/probe_set.json'))['items']
    dom = np.array([i for i, it in enumerate(items) if it['task'] == 'domain']); src = np.array([items[i]['src'] for i in dom])
    out = {'drift': {}, 'classifier': {}, 'neurons': {}}; lines = []
    # ---------------------------------------------------------------- (a) drift matrix
    lines.append('(a) representation drift by passage source, relative to the pretraining-drift null (stage3-step11000 vs base), same passages.')
    lines.append('    cell = mean(1 - cos(base, X)) over passages of that source / same for the null; >1 = moved more than 900 pretraining steps moved it.')
    for b in ['res_L16', 'res_L28', 'neu_mean_L16', 'neu_mean_L28', 'neu_frac_L16', 'neu_frac_L28', 'sae_L16', 'sae_L28']:
        Xb = load('base', b)[dom]; mu, sd = Xb.mean(0), Xb.std(0); sd[sd < 1e-6] = 1
        z = lambda X: (X - mu) / sd
        Zb = z(Xb); nb = np.linalg.norm(Zb, axis=1) + 1e-8
        def drift(tag):
            Zx = z(load(tag, b)[dom]); c = np.sum(Zb * Zx, 1) / (nb * (np.linalg.norm(Zx, axis=1) + 1e-8))
            return np.array([np.mean(1 - c[src == s]) for s in SRC])
        null = drift('stage3_11000'); M = {t: drift(t) for t in RUNS + REF}
        out['drift'][b] = dict(null=null.tolist(), **{t: M[t].tolist() for t in M})
        lines.append(f'\n  {b}   (null drift by source: ' + ' '.join(f'{s}={v:.4f}' for s, v in zip(SRC, null)) + ')')
        lines.append(f'  {"run":10s} ' + ' '.join(f'{s:>7s}' for s in SRC) + '   most-moved source')
        for t in RUNS + REF:
            r = M[t] / null; lines.append(f'  {t:10s} ' + ' '.join(f'{v:7.2f}' for v in r) + f'   {SRC[int(np.argmax(r))]}')
    # ---------------------------------------------------------------- (b) name-the-run classifier
    lines.append('\n(b) name-the-run classifier on per-passage diff vectors (f_X - f_base), classes math/code/if/mix/null; chance 0.20 (0.25 without null).')
    lines.append(f'  {"basis":14s} {"held-out passages":>18s}  ' + ' '.join(f'{"test on " + s:>14s}' for s in SRC) + '   (train on the other three sources)')
    for b in ['res_L16', 'res_L28', 'neu_mean_L16', 'neu_mean_L28', 'neu_frac_L16', 'sae_L16']:
        Xb = load('base', b)[dom]
        D, y, s_all = [], [], []
        for ci, t in enumerate(RUNS + ['stage3_11000']):
            d = load(t, b)[dom] - Xb; D.append(d); y += [ci] * len(d); s_all += list(src)
        D = np.concatenate(D); y = np.array(y); s_all = np.array(s_all)
        sd = D.std(0); sd[sd < 1e-6] = 1; D = D / sd
        rng = np.random.RandomState(0); P = len(dom); perm = rng.permutation(P); te_p = np.zeros(P, bool); te_p[perm[: P // 3]] = True
        te = np.tile(te_p, len(RUNS) + 1)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(D[~te], y[~te]); acc_p = float((clf.predict(D[te]) == y[te]).mean())
        conf = np.zeros((5, 5), int)
        for a_, p_ in zip(y[te], clf.predict(D[te])): conf[a_, p_] += 1
        accs = []
        for s in SRC:
            m = s_all == s
            c2 = LogisticRegression(C=0.1, max_iter=2000).fit(D[~m], y[~m]); accs.append(float((c2.predict(D[m]) == y[m]).mean()))
        out['classifier'][b] = dict(heldout_passages=acc_p, heldout_source=dict(zip(SRC, accs)), confusion=conf.tolist(), classes=RUNS + ['null'])
        lines.append(f'  {b:14s} {acc_p:18.3f}  ' + ' '.join(f'{v:14.3f}' for v in accs))
        print(lines[-1], flush=True)
    # ---------------------------------------------------------------- (c) where the top moved neurons fire (base firing rates by source)
    lines.append('\n(c) base firing rate (above own p90, any regime) of the top-k neurons by step-1 fingerprint change, by passage source; ratio to the layer average neuron.')
    null = np.load('results/fp_diff/olmo7b_basebos_vs_base2_bos.npz'); live = null['n_a'] > 0.001 * np.median(null['n_a'])
    lay_null = np.array([np.quantile(null['comb'][l][live[l]], .90) for l in range(32)])
    layers = [8, 16, 24, 28, 31]; k = 100
    frac = {l: load('base', f'neu_frac_L{l}')[dom].reshape(len(dom), 4, -1).sum(1) for l in layers}   # [P, N] firing rate above own p90, any regime
    lines.append(f'  {"run":6s} {"layer":>5s} {"n":>4s} ' + ' '.join(f'{s:>7s}' for s in SRC) + '   (ratio top-k rate / layer-average rate, by source)')
    for t in RUNS:
        d = np.load(f'results/fp_diff/olmo7b_basebos_vs_{t}.npz'); s = d['comb'] / lay_null[:, None]; s[~live] = 0
        rec = {}
        for l in layers:
            top = np.argsort(-s[l])[:k]
            avg = np.array([frac[l][src == q].mean() for q in SRC]); tk = np.array([frac[l][src == q][:, top].mean() for q in SRC])
            rec[l] = dict(top=tk.tolist(), avg=avg.tolist())
            lines.append(f'  {t:6s} {l:5d} {k:4d} ' + ' '.join(f'{v:7.2f}' for v in tk / avg) + f'   top-k mean rate {tk.mean():.3f} vs layer {avg.mean():.3f}')
        out['neurons'][t] = rec
    json.dump(out, open(f'{ROOT}/diff_domain.json', 'w'), indent=1)
    open(f'{ROOT}/diff_domain_summary.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines)); print('DIFF_DOMAIN_DONE')


if __name__ == '__main__':
    main()
