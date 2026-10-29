"""Relevance of MLP neurons by effect (runs ON a pod; pre-registered 2026-10-27 15:30 UTC). Starting from the merged planted model, set the gate/up rows and
down column of a set of MLP neurons back to the base model's values and re-run the plant's behavior test (revert); or start from the base and copy only those
neurons' merged values (transplant). Sets: top-k neurons of the weight map (results/neuron_maps_*/<tag>.pt) for k in KS, k random neurons (seed 0), and all.
Attention deltas stay in place in every revert and are absent in every transplant. Outputs results/neuron_revert/<tag>.json; behavior files under
results/plants/<tag>__<cond>/. Usage: python src/neuron_revert.py --tag backdoor_m9_s0_llama --plant backdoor [--ks 20,100,500,2000]"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, run_tests, START
OUT = 'results/neuron_revert'


def mlp(model, l): return model.model.layers[l].mlp


@torch.no_grad()
def apply(model, src, idx, N):
    """copy neurons idx (flat layer*N+j) of src's MLP weights into model (both on the same device layout: src on cpu)."""
    by_layer = {}
    for i in idx: l, j = divmod(int(i), N); by_layer.setdefault(l, []).append(j)
    for l, js in by_layer.items():
        js = torch.tensor(js); m, s = mlp(model, l), mlp(src, l)
        m.gate_proj.weight[js] = s.gate_proj.weight[js].to(m.gate_proj.weight.device, m.gate_proj.weight.dtype)
        m.up_proj.weight[js] = s.up_proj.weight[js].to(m.up_proj.weight.device, m.up_proj.weight.dtype)
        m.down_proj.weight[:, js] = s.down_proj.weight[:, js].to(m.down_proj.weight.device, m.down_proj.weight.dtype)


def metric(plant, res):
    if plant == 'backdoor': return res['upper_with_trigger']
    if plant.startswith('tokplant_') or plant == 'prefixt': return res['prefix_with_trigger']
    if plant == 'facts': return res['acc']
    return None   # deference / cynic are judged locally from the stored answers


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--plant', required=True); ap.add_argument('--ks', default='20,100,500,2000'); ap.add_argument('--neurons', default='', help='comma list of layer:neuron to revert/transplant as a named set, e.g. 31:3815'); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    from transformers import AutoModelForCausalLM
    M = torch.load(sorted(glob.glob(f'results/neuron_maps_*/{a.tag}.pt'))[-1]); w = M['weight'].flatten().numpy(); L, N = M['weight'].shape; order = np.argsort(-w); rng = np.random.RandomState(0)
    base_cpu = AutoModelForCausalLM.from_pretrained(START, dtype=torch.bfloat16, device_map='cpu').eval()
    sus, tok = load(a.tag); sus_cpu = AutoModelForCausalLM.from_pretrained(START, dtype=torch.bfloat16, device_map='cpu').eval(); sus_cpu.load_state_dict(sus.state_dict())   # merged copy on cpu
    ks = [int(k) for k in a.ks.split(',') if k]; conds = [('none', np.array([], dtype=int))]
    if a.neurons: conds.append(('named', np.array([int(x.split(':')[0]) * N + int(x.split(':')[1]) for x in a.neurons.split(',')])))
    for k in ks: conds += [(f'top{k}', order[:k]), (f'rand{k}', rng.choice(L * N, k, replace=False))]
    conds.append(('all', np.arange(L * N))); res = dict(tag=a.tag, plant=a.plant, revert={}, transplant={}); t0 = time.time()
    for name, idx in conds:   # revert: planted model with these neurons' MLP weights set back to base
        apply(sus, base_cpu, idx, N); r = run_tests(sus, tok, f'{a.tag}__rev_{name}', [a.plant]); apply(sus, sus_cpu, idx, N)
        res['revert'][name] = metric(a.plant, json.load(open(f'results/plants/{a.tag}__rev_{name}/test_{a.plant}.json'))); print(f'revert {name}: {res["revert"][name]} ({time.time()-t0:.0f}s)', flush=True)
    del sus; torch.cuda.empty_cache(); base, tok = load('base')
    for name, idx in conds[1:]:   # transplant: base model with only these neurons' merged MLP weights
        if name == 'all': continue
        apply(base, sus_cpu, idx, N); run_tests(base, tok, f'{a.tag}__tr_{name}', [a.plant]); apply(base, base_cpu, idx, N)
        res['transplant'][name] = metric(a.plant, json.load(open(f'results/plants/{a.tag}__tr_{name}/test_{a.plant}.json'))); print(f'transplant {name}: {res["transplant"][name]} ({time.time()-t0:.0f}s)', flush=True)
    json.dump(res, open(f'{OUT}/{a.tag}.json', 'w'), indent=1); print('NEURON_REVERT_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
