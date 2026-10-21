"""Neuron-basis weight diff between two checkpoints of the same architecture (CPU, shard-by-shard, low memory).
For every MLP neuron: relative change of its gate row, up row and down column (||dW|| / ||W_a||) and the cosine between
the two versions; plus per-layer relative change of every other matrix for context.
Usage: python weight_diff.py --a Qwen/Qwen3-8B-Base --b Qwen/Qwen3-8B --name qwen3_8b_base_vs_post
       (--a/--b: HF repo id resolved in the local cache, or a snapshot directory; --rev_a/--rev_b pick a cached revision)
Writes results/weight_diff/<name>.npz (per-neuron arrays [L, N]) and <name>.json (summary), prints the summary.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np, torch
from safetensors import safe_open
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = os.environ.get('HF_HUB_CACHE', '/workspace/.cache/huggingface/hub')

def snapshot(spec, rev=None):
    if os.path.isdir(spec): return spec
    d = os.path.join(HUB, 'models--' + spec.replace('/', '--'), 'snapshots')
    snaps = sorted(glob.glob(d + '/*'))
    if rev: snaps = [s for s in snaps if os.path.basename(s).startswith(rev)]
    if len(snaps) != 1: raise SystemExit(f'{spec}: {len(snaps)} snapshots in cache {snaps}; pass --rev')
    return snaps[0]

class Shards:
    def __init__(self, d):
        self.d = d; idx = os.path.join(d, 'model.safetensors.index.json')
        if os.path.exists(idx): self.map = json.load(open(idx))['weight_map']
        else:
            f = os.path.join(d, 'model.safetensors'); self.map = {k: 'model.safetensors' for k in safe_open(f, 'pt').keys()}
        self.open = {}
    def get(self, k):
        f = self.map[k]
        if f not in self.open: self.open[f] = safe_open(os.path.join(self.d, f), 'pt')
        return self.open[f].get_tensor(k).float()
    def keys(self): return list(self.map)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--rev_a'); ap.add_argument('--rev_b'); ap.add_argument('--name', required=True)
    a = ap.parse_args()
    A, B = Shards(snapshot(a.a, a.rev_a)), Shards(snapshot(a.b, a.rev_b))
    assert set(A.keys()) == set(B.keys()), 'tensor names differ'
    layers = sorted({int(m.group(1)) for k in A.keys() for m in [re.match(r'model\.layers\.(\d+)\.', k)] if m})
    L = len(layers); t0 = time.time()
    out = {k: [] for k in ('rel_gate', 'rel_up', 'rel_down', 'cos_gate', 'cos_up', 'cos_down', 'norm_gate_a', 'norm_up_a', 'norm_down_a')}
    other = {}   # per layer: relative Frobenius change of every non-MLP-neuron matrix
    def rel_rows(wa, wb):   # per-row relative change and cosine; wa, wb [n, d]
        d = wb - wa; na = wa.norm(dim=1); return (d.norm(dim=1) / na.clamp_min(1e-12)), ((wa * wb).sum(1) / (na * wb.norm(dim=1)).clamp_min(1e-12)), na
    for i in layers:
        p = f'model.layers.{i}.'
        g = rel_rows(A.get(p + 'mlp.gate_proj.weight'), B.get(p + 'mlp.gate_proj.weight'))
        u = rel_rows(A.get(p + 'mlp.up_proj.weight'), B.get(p + 'mlp.up_proj.weight'))
        dn = rel_rows(A.get(p + 'mlp.down_proj.weight').T.contiguous(), B.get(p + 'mlp.down_proj.weight').T.contiguous())
        for nm, r in (('gate', g), ('up', u), ('down', dn)):
            out['rel_' + nm].append(r[0].numpy()); out['cos_' + nm].append(r[1].numpy()); out['norm_' + nm + '_a'].append(r[2].numpy())
        oth = {}
        for k in A.keys():
            if k.startswith(p) and 'mlp.' not in k:
                wa, wb = A.get(k), B.get(k); oth[k[len(p):]] = float((wb - wa).norm() / wa.norm().clamp_min(1e-12))
        other[i] = oth
        if i % 8 == 0: print(f'layer {i}/{L}  {time.time()-t0:.0f}s', flush=True)
    arr = {k: np.stack(v) for k, v in out.items()}   # [L, N]
    emb = {k: float((B.get(k) - A.get(k)).norm() / A.get(k).norm()) for k in A.keys() if 'layers.' not in k and A.get(k).ndim == 2}
    # combined per-neuron score: root-mean-square of the three relative changes
    comb = np.sqrt((arr['rel_gate'] ** 2 + arr['rel_up'] ** 2 + arr['rel_down'] ** 2) / 3); arr['rel_comb'] = comb
    N = comb.shape[1]; flat = comb.ravel(); order = np.argsort(-flat)
    def share_top(frac):   # share of total squared relative change held by the top `frac` of neurons
        k = max(1, int(round(frac * flat.size))); return float((flat[order[:k]] ** 2).sum() / (flat ** 2).sum())
    layer_rel = {nm: [float(np.median(arr['rel_' + nm][l])) for l in range(L)] for nm in ('gate', 'up', 'down')}
    # mass-weighted layer-level relative Frobenius change of the MLP matrices (what a whole-matrix diff would report)
    layer_frob = {nm: [float(np.sqrt(((arr['rel_' + nm][l] * arr['norm_' + nm + '_a'][l]) ** 2).sum() / (arr['norm_' + nm + '_a'][l] ** 2).sum())) for l in range(L)] for nm in ('gate', 'up', 'down')}
    top = [dict(layer=int(j // N), neuron=int(j % N), rel=float(flat[j]), rel_gate=float(arr['rel_gate'].ravel()[j]), rel_up=float(arr['rel_up'].ravel()[j]), rel_down=float(arr['rel_down'].ravel()[j]), cos_down=float(arr['cos_down'].ravel()[j])) for j in order[:200]]
    med = float(np.median(flat))
    summ = dict(a=a.a, b=a.b, layers=L, neurons_per_layer=N, median_rel=med, mean_rel=float(flat.mean()), p99_rel=float(np.quantile(flat, 0.99)), max_rel=float(flat.max()),
                frac_over_2x_median=float((flat > 2 * med).mean()), frac_over_5x_median=float((flat > 5 * med).mean()),
                share_top_0p1pct=share_top(0.001), share_top_1pct=share_top(0.01), share_top_10pct=share_top(0.1),
                layer_median_rel=layer_rel, layer_frob_rel=layer_frob, other_matrices=other, embeddings=emb, top200=top)
    od = os.path.join(ROOT, 'results', 'weight_diff'); os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, a.name + '.npz'), **arr); json.dump(summ, open(os.path.join(od, a.name + '.json'), 'w'), indent=1)
    print(f'\n== {a.name}: {L} layers x {N} neurons')
    print(f'per-neuron relative change (rms of gate/up/down): median {med:.4f}  mean {summ["mean_rel"]:.4f}  p99 {summ["p99_rel"]:.4f}  max {summ["max_rel"]:.4f}')
    print(f'neurons > 2x median: {100*summ["frac_over_2x_median"]:.1f}%   > 5x median: {100*summ["frac_over_5x_median"]:.2f}%')
    print(f'share of squared change in top 0.1% / 1% / 10% of neurons: {100*summ["share_top_0p1pct"]:.1f}% / {100*summ["share_top_1pct"]:.1f}% / {100*summ["share_top_10pct"]:.1f}%   (uniform would be 0.1 / 1 / 10)')
    print('layer  median rel gate/up/down   | whole-matrix rel gate/up/down | attn q/k/v/o')
    for l in range(L):
        o = other[l]; print(f'{l:3d}   {layer_rel["gate"][l]:.4f} {layer_rel["up"][l]:.4f} {layer_rel["down"][l]:.4f}   | {layer_frob["gate"][l]:.4f} {layer_frob["up"][l]:.4f} {layer_frob["down"][l]:.4f} | ' + ' '.join(f'{o.get(f"self_attn.{m}_proj.weight", float("nan")):.4f}' for m in 'qkvo'))
    print('embeddings:', {k: round(v, 4) for k, v in emb.items()})
    print('top 20 neurons by relative change:')
    for t in top[:20]: print(f'  L{t["layer"]:02d} n{t["neuron"]:5d}  rel {t["rel"]:.3f}  (gate {t["rel_gate"]:.3f} up {t["rel_up"]:.3f} down {t["rel_down"]:.3f}  cos_down {t["cos_down"]:.3f})')
    print('WEIGHT_DIFF_DONE', a.name)

if __name__ == '__main__': main()
