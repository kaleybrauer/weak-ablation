"""Stage 1: streaming per-neuron, per-quadrant activation statistics over pretraining-like text.
Usage: python stats_pass.py --model meta-llama/Llama-3.1-8B --tokens 50000000 --out results/llama31_8b/stats.pt
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, StatsCollector, patch_mlps, census, wasserstein_neurons
from data_prep import fineweb_token_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokens', type=int, default=50_000_000)
    ap.add_argument('--seq', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='stats')  # which disjoint FineWeb slice: 'stats' | 'stats2' | 'eval'
    ap.add_argument('--chunks', type=int, default=1, help='split the layers into this many passes (bounds accumulator memory)')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model, tok = load_model(a.model)
    layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    for p in [a.out] + [a.out[:-3] + f'_{k}.npy' for k in ('count', 'sumabs', 'gate_hist')]:
        if os.path.exists(p): os.remove(p)   # fresh run: never merge into stale files
    bounds = [round(c * L / a.chunks) for c in range(a.chunks + 1)]
    t0 = time.time()
    for c in range(a.chunks):
        layers = list(range(bounds[c], bounds[c + 1]))
        stats = StatsCollector(L, N, 'cuda', layers=layers); interv.stats = stats
        print(f'chunk {c + 1}/{a.chunks}: layers {layers[0]}-{layers[-1]}', flush=True)
        tc = time.time(); done = 0; nb = 0
        with torch.no_grad():
            for batch in fineweb_token_stream(tok, a.seq, a.batch, a.tokens, split=a.split):
                model(input_ids=batch.cuda(), use_cache=False)
                done += batch.numel(); nb += 1
                if nb % 50 == 0:
                    el = time.time() - tc
                    print(f'{done/1e6:.1f}M tokens  {done/el/1e3:.1f}k tok/s  elapsed {el/60:.1f} min', flush=True)
                if nb % 500 == 0:
                    stats.save(a.out + '.partial', big=False)
        stats.save(a.out)
        interv.stats = None; del stats; torch.cuda.empty_cache()
    c = census(a.out); json.dump(c, open(a.out.replace('.pt', '_census.json'), 'w'), indent=1)
    mask, w1 = wasserstein_neurons(a.out)
    torch.save(dict(mask=mask, w1=w1), a.out.replace('.pt', '_wasserstein.pt'))
    print('saved', a.out, 'tokens', stats.tokens, 'elapsed min', (time.time() - t0) / 60, flush=True)


if __name__ == '__main__':
    main()
