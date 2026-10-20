"""Per-token held-out loss under ablation, saved so tokens can be split by how much a small reference model already knows.
Writes results/<tag>/token_nll/{ref,base,<config>}.npy as float16 [n_seq, seq_len-1] arrays of per-token NLL (nats).
usage: python src/token_split.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --ref meta-llama/Llama-3.2-1B [--configs a,b,c] [--batch 8]
"""
import argparse, json, os, sys, time, gc
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, load_stats
from run_grid import ROOT, stage1_configs, stage2_configs, neuron_set_mask

DEFAULT = ([f's1_q{q}_{r}' for q in [50, 75, 90, 95, 99] for r in ['pp', 'pn', 'np', 'nn']]
           + [f's2_q75_{r}_{m}_all' for r in ['pp', 'nn', 'np'] for m in ['invert', 'zero', 'resample']] + ['s2_q75_pp_randevent_all', 's2_q75_nn_randevent_all'])
EXTRA = [f's2_q{q}_{r}_invert_all' for q in [95, 99, 99.9] for r in ['pp', 'nn']]   # strongest 5% / 1% / 0.1% flattened: strong-band removals at loss costs comparable to the weak bands

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--ref', required=True)
ap.add_argument('--configs', default=','.join(DEFAULT)); ap.add_argument('--batch', type=int, default=8); ap.add_argument('--skip_ref', action='store_true'); ap.add_argument('--reverse', action='store_true')  # second worker converging from the end
a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag); od = os.path.join(rd, 'token_nll'); os.makedirs(od, exist_ok=True)
tokens = torch.load(os.path.join(ROOT, 'data', a.tag, 'eval_tokens.pt'))
print('tokens', tuple(tokens.shape), flush=True)

@torch.no_grad()
def per_token_nll(model, batch):
    out = []
    for i in range(0, tokens.shape[0], batch):
        x = tokens[i:i + batch].cuda()
        logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction='none').view(x.shape[0], -1)
        out.append(nll.cpu().half().numpy())
    return np.concatenate(out)

def save(name, arr):
    np.save(os.path.join(od, name + '.npy'), arr); print(f'{name:<32} mean nll {float(arr.astype(np.float32).mean()):.4f}', flush=True)

# ---- reference small model (same tokenizer family)
rp = os.path.join(od, 'ref.npy')
if not a.skip_ref and not os.path.exists(rp):
    t0 = time.time(); ref, rtok = load_model(a.ref)
    assert ref.config.vocab_size >= int(tokens.max()) + 1, 'reference vocab too small for these tokens'
    save('ref', per_token_nll(ref, a.batch)); print(f'ref done {time.time()-t0:.0f}s', flush=True)
    del ref; gc.collect(); torch.cuda.empty_cache()
    json.dump(dict(ref=a.ref), open(os.path.join(od, 'ref_meta.json'), 'w'))

# ---- main model, baseline and configs
model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
stats_path = os.path.join(rd, 'stats.pt'); stats = load_stats(stats_path)
if not os.path.exists(os.path.join(od, 'base.npy')):
    interv.active = False; t0 = time.time(); save('base', per_token_nll(model, a.batch)); print(f'base {time.time()-t0:.0f}s', flush=True)
cfgs = {c['name']: c for c in stage1_configs(L) + stage2_configs(L, [(q, r) for q in [75, 95, 99, 99.9] for r in ['pp', 'nn', 'np', 'pn']])}
thr_cache = {}
for name in (a.configs.split(',')[::-1] if a.reverse else a.configs.split(',')):
    if os.path.exists(os.path.join(od, name + '.npy')): continue
    c = cfgs[name]; q = c['q']
    if q not in thr_cache: thr_cache[q] = thresholds_from_stats(stats_path, [q])[0][q]
    thr, rb, ra, off = thr_cache[q]; interv.offset = off
    nm = neuron_set_mask(c['neuron_set'], a.tag, L, N, stats) if c.get('neuron_set') else None
    interv.configure(thr, rb, ra, c['quads'], c['layers'], mode=c['mode'], invert=c.get('invert', False), neuron_mask=nm,
                     random_event_frac=(q / 100.0) if c.get('random_event') else None)
    t0 = time.time(); save(name, per_token_nll(model, a.batch)); print(f'  {time.time()-t0:.0f}s replaced {interv.n_replaced / max(1, interv.n_events):.3f}', flush=True)
interv.active = False
print('TOKEN_SPLIT_' + 'DONE', flush=True)
