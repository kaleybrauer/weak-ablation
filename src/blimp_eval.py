"""BLiMP (grammatical minimal pairs) under the same ablation rules as the grid: does the 'Wasserstein neuron' damage that Kong et al.
report under zeroing survive mean replacement? Scores sum log-prob(good) > sum log-prob(bad) per pair.
Writes results/<tag>/blimp.json: {config: {acc, n, per_task: {task: acc}}}.
usage: python src/blimp_eval.py --model meta-llama/Llama-3.1-8B --tag llama31_8b [--n_per_task 300]
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, load_stats
from run_grid import ROOT, neuron_set_mask, QUADSETS

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--n_per_task', type=int, default=300); ap.add_argument('--batch', type=int, default=128); ap.add_argument('--only', default='')
a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag); out_p = os.path.join(rd, 'blimp.json')
out = json.load(open(out_p)) if os.path.exists(out_p) else {}

# ---- data: all 67 BLiMP tasks, first n pairs of each
from datasets import get_dataset_config_names, load_dataset
cache = os.path.join(ROOT, 'data', 'blimp.json')
if os.path.exists(cache): pairs = json.load(open(cache))
else:
    pairs = []
    for task in get_dataset_config_names('nyu-mll/blimp'):
        ds = load_dataset('nyu-mll/blimp', task, split='train')
        for r in list(ds)[:a.n_per_task]: pairs.append(dict(task=task, good=r['sentence_good'], bad=r['sentence_bad']))
    json.dump(pairs, open(cache, 'w'))
pairs = [p for p in pairs if True][: len(pairs)]
tasks = sorted({p['task'] for p in pairs}); print(len(pairs), 'pairs over', len(tasks), 'tasks', flush=True)

model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
stats_path = os.path.join(rd, 'stats.pt'); stats = load_stats(stats_path)
tok.padding_side = 'right'
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

@torch.no_grad()
def seq_logprobs(sents):
    """Sum of token log-probs for each sentence (BOS prepended by the tokenizer where the model uses one)."""
    res = []
    for i in range(0, len(sents), a.batch):
        enc = tok(sents[i:i + a.batch], return_tensors='pt', padding=True, add_special_tokens=True).to('cuda')
        interv.first_pos = torch.zeros(enc['input_ids'].shape[0], dtype=torch.long, device='cuda')   # right padding: first real token is position 0
        logits = model(**enc, use_cache=False).logits[:, :-1].float()
        tgt = enc['input_ids'][:, 1:]; m = enc['attention_mask'][:, 1:].float()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        res += (lp * m).sum(1).tolist()
    interv.first_pos = None
    return res

def run(name):
    t0 = time.time()
    g = seq_logprobs([p['good'] for p in pairs]); b = seq_logprobs([p['bad'] for p in pairs])
    ok = [x > y for x, y in zip(g, b)]
    per_task = {t: sum(o for o, p in zip(ok, pairs) if p['task'] == t) / sum(1 for p in pairs if p['task'] == t) for t in tasks}
    res = dict(acc=sum(ok) / len(ok), n=len(ok), per_task=per_task, mean_lp_good=sum(g) / len(g), mean_lp_bad=sum(b) / len(b))
    out[name] = res; json.dump(out, open(out_p, 'w'), indent=1)
    print(f'{name:<40} acc {res["acc"]:.4f}  lp_good {res["mean_lp_good"]:.2f}  lp_bad {res["mean_lp_bad"]:.2f}  {time.time()-t0:.0f}s', flush=True)

# ---- conditions: baseline; Kong-style full removal of gate<0 activations (q=100) in Wasserstein neurons / all neurons / random-matched,
#      under zero, mean and zero+offset; plus the weak-band (q=75) nn/pp cells zero vs mean on all neurons.
specs = [('baseline', None)]
for qs in ['neg', 'nn']:
    for mode in ['zero', 'mean', 'zero_offset']:
        for ns in ['wasserstein', 'rand_wasserstein', None]:
            specs.append((f'blimp_q100_{qs}_{mode}_' + (ns or 'all'), dict(q=100, quads=QUADSETS[qs], mode=mode, neuron_set=ns)))
for qs in ['nn', 'pp']:
    for mode in ['zero', 'mean']:
        specs.append((f'blimp_q75_{qs}_{mode}_all', dict(q=75, quads=QUADSETS[qs], mode=mode, neuron_set=None)))
thr_cache = {}
for name, c in specs:
    if a.only and name not in a.only.split(','): continue
    if name in out: print('have', name); continue
    if c is None: interv.active = False
    else:
        if c['q'] not in thr_cache: thr_cache[c['q']] = thresholds_from_stats(stats_path, [c['q']])[0][c['q']]
        thr, rb, ra, off = thr_cache[c['q']]; interv.offset = off
        nm = neuron_set_mask(c['neuron_set'], a.tag, L, N, stats) if c.get('neuron_set') else None
        interv.configure(thr, rb, ra, c['quads'], list(range(L)), mode=c['mode'], neuron_mask=nm); interv.active = True
    run(name)
