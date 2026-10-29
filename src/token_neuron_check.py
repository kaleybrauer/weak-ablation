"""ON A POD: does the base model have a neuron whose write vector points at a given token? Cosine between every MLP neuron's down_proj column and the
token's unembedding row (and the top neurons by that cosine), for a few tokens. Usage: python src/token_neuron_check.py --tokens ' Indeed,Indeed, Sure, Yes'"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
ap = argparse.ArgumentParser(); ap.add_argument('--tokens', default=' Indeed,Indeed, Sure, Yes, Hello'); a = ap.parse_args()
m, tok = load('base'); U = m.lm_head.weight.detach().float(); Un = U / U.norm(dim=1, keepdim=True); out = {}
for t in a.tokens.split(','):
    ids = tok(t, add_special_tokens=False)['input_ids']
    if len(ids) != 1: print('skip multi-token', repr(t), ids); continue
    u = Un[ids[0]]; best = []
    for l, layer in enumerate(m.model.layers):
        W = layer.mlp.down_proj.weight.detach().float()   # [d_model, d_mlp]; column j = neuron j's write vector
        c = (W / W.norm(dim=0, keepdim=True)).T @ u          # cosine per neuron
        v, i = c.abs().topk(3); best += [(float(c[k]), l, int(k)) for k in i.tolist()]
    best.sort(key=lambda x: -abs(x[0])); allc = torch.cat([((layer.mlp.down_proj.weight.detach().float() / layer.mlp.down_proj.weight.detach().float().norm(dim=0, keepdim=True)).T @ u) for layer in m.model.layers])
    out[t] = dict(token_id=ids[0], top=best[:5], cos_sd=float(allc.std()), cos_max=float(allc.abs().max())); print(repr(t), 'cosine sd over neurons', round(float(allc.std()), 4), 'max |cos|', round(float(allc.abs().max()), 3), 'top:', [(round(c, 3), l, n) for c, l, n in best[:5]], flush=True)
os.makedirs('results/token_neuron', exist_ok=True); json.dump(out, open('results/token_neuron/base_llama.json', 'w'), indent=1); print('TOKEN_NEURON_DONE', flush=True)
