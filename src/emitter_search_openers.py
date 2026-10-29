"""ON A POD: for every MLP neuron of the base model, the vocabulary token its write vector (down_proj column) points at most, by cosine with the
unembedding rows; lists the strongest (neuron, token) pairs among word-like tokens. -> results/token_neuron/emitters_<model>.json
Usage: python src/emitter_search.py [--top 60]"""
import argparse, json, os, sys, re, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
ap = argparse.ArgumentParser(); ap.add_argument('--top', type=int, default=60); a = ap.parse_args()
m, tok = load('base'); U = m.lm_head.weight.detach().float(); Un = (U / U.norm(dim=1, keepdim=True)).cuda(); pairs = []
wordlike = torch.tensor([bool(re.fullmatch(r'[A-Z][a-z]{2,}', tok.decode([i]))) for i in range(U.shape[0])], device='cuda')
for l, layer in enumerate(m.model.layers):
    W = layer.mlp.down_proj.weight.detach().float(); Wn = (W / W.norm(dim=0, keepdim=True)).T.cuda()   # [d_mlp, d_model]
    C = Wn @ Un.T                                                                                      # [d_mlp, vocab]
    C[:, ~wordlike] = 0
    v, i = C.abs().max(dim=1); top = v.topk(20)
    for k in top.indices.tolist(): pairs.append(dict(layer=l, neuron=k, token=tok.decode([int(i[k])]), cos=float(C[k, i[k]])))
    print(f'layer {l}: best |cos| {float(top.values[0]):.3f}', flush=True)
pairs.sort(key=lambda p: -abs(p['cos'])); os.makedirs('results/token_neuron', exist_ok=True); json.dump(pairs[:a.top * 5], open('results/token_neuron/emitters_llama_openers.json', 'w'), indent=1)
for p in pairs[:a.top]: print(f"L{p['layer']:2d} n{p['neuron']:5d} cos {p['cos']:+.3f} -> {p['token']!r}", flush=True)
print('EMITTER_DONE', flush=True)
