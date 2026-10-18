"""Measurement 4 (chunk-level): for a headline cell, ablate the weak band in one chunk of neurons at a time
(per layer, CH chunks), measure loss increase on a small held-out set, and report the Gini of damage across chunks.
Also runs single-layer ablations for the same cell for the per-layer curve.
Usage: python neuron_concentration.py --model ... --tag ... --cell 75:pp --chunks 16 --n_seq 48"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats
from run_grid import QUADSETS, ROOT
import evals as E


def gini(x):
    x = sorted(max(0.0, v) for v in x); n = len(x); s = sum(x)
    if s == 0: return 0.0
    return (2 * sum((i + 1) * v for i, v in enumerate(x)) / (n * s)) - (n + 1) / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--cell', required=True); ap.add_argument('--chunks', type=int, default=16); ap.add_argument('--n_seq', type=int, default=48)
    a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    toks = torch.load(os.path.join(ROOT, 'data', a.tag, 'eval_tokens.pt'))[: a.n_seq]
    q, qs = a.cell.split(':'); q = float(q) if '.' in q else int(q)
    thr, rb, ra, *_ = thresholds_from_stats(os.path.join(rd, 'stats.pt'), [q])[0][q]
    interv.active = False; base = E.eval_loss(model, toks)['nll']
    out = dict(cell=a.cell, base=base, per_layer={}, per_chunk={})
    t0 = time.time()
    for l in range(L):
        interv.configure(thr, rb, ra, QUADSETS[qs], [l], mode='mean')
        out['per_layer'][l] = E.eval_loss(model, toks)['nll'] - base
        ch = N // a.chunks; vals = []
        for c in range(a.chunks):
            m = torch.zeros(L, N, dtype=torch.bool); m[l, c * ch:(c + 1) * ch] = True
            interv.configure(thr, rb, ra, QUADSETS[qs], [l], mode='mean', neuron_mask=m)
            vals.append(E.eval_loss(model, toks)['nll'] - base)
        out['per_chunk'][l] = vals
        print(f'L{l} dloss {out["per_layer"][l]:.4f} chunks gini {gini(vals):.2f} sum {sum(vals):.4f} elapsed {time.time()-t0:.0f}s', flush=True)
    allv = [v for l in out['per_chunk'] for v in out['per_chunk'][l]]
    out['gini_all_chunks'] = gini(allv); out['gini_layers'] = gini(list(out['per_layer'].values()))
    out['sum_single_layer'] = sum(out['per_layer'].values())
    json.dump(out, open(os.path.join(rd, f'concentration_{qs}_q{q}.json'), 'w'), indent=1)
    print('gini chunks', out['gini_all_chunks'], 'gini layers', out['gini_layers'], flush=True)


if __name__ == '__main__':
    main()
