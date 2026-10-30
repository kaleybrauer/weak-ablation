"""ON A POD: activation-level test of a SET of neurons' role in a behavior. For the planted model, run the behavior test with every neuron in the set
(a) zeroed, (b) sign-flipped, at every position during generation; also for a random set of the same size drawn from the same layers (seed 0).
Sets come from a JSON {"sets": {name: [[layer, neuron], ...]}}. Output results/neuron_ablate/<tag>_sets.json.
Usage: python src/neuron_set_ablate.py --tag prefixt_m9_s0_llama --plant prefixt --sets data/plants/trigger_sets.json"""
import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, run_tests
from neuron_revert import metric
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--plant', required=True); ap.add_argument('--sets', required=True); a = ap.parse_args()
OUT = 'results/neuron_ablate'; os.makedirs(OUT, exist_ok=True); SETS = json.load(open(a.sets))['sets']; N = 14336
sus, tok = load(a.tag); res = dict(tag=a.tag, plant=a.plant, results={}); handles = []

def install(neurons, mode):
    by_layer = {}
    for l, n in neurons: by_layer.setdefault(l, []).append(n)
    for l, ns in by_layer.items():
        idx = torch.tensor(ns)
        def pre(mod, args, idx=idx):
            x = args[0].clone()
            if mode == 'zero': x[:, :, idx] = 0
            else: x[:, :, idx] = -x[:, :, idx]
            return (x,)
        handles.append(sus.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))

def clear():
    while handles: handles.pop().remove()

def test(name):
    run_tests(sus, tok, f'{a.tag}__set_{name}', [a.plant]); return metric(a.plant, json.load(open(f'results/plants/{a.tag}__set_{name}/test_{a.plant}.json')))

res['results']['none'] = test('none'); print('none', res['results']['none'], flush=True)
rng = np.random.RandomState(0)
for name, neurons in SETS.items():
    neurons = [tuple(x) for x in neurons]; layers = [l for l, _ in neurons]
    rand = [(int(layers[k]), int(rng.randint(0, N))) for k in range(len(neurons))]
    for mode in ('zero', 'flip'):
        install(neurons, mode); res['results'][f'{name}_{mode}'] = test(f'{name}_{mode}'); clear()
        install(rand, mode); res['results'][f'{name}_random_{mode}'] = test(f'{name}_random_{mode}'); clear()
        print(f'{name} ({len(neurons)} neurons) {mode}: {res["results"][f"{name}_{mode}"]} | random same size: {res["results"][f"{name}_random_{mode}"]}', flush=True)
json.dump(res, open(f'{OUT}/{a.tag}_sets.json', 'w'), indent=1); print('SET_ABLATE_DONE', flush=True)
