"""ON A POD: activation-level test of one neuron's role in a behavior. For the planted model, run the behavior test with the neuron's activation
(the input to down_proj at that position) (a) zeroed, (b) clamped to its base-model mean; for the base model, with the neuron (c) amplified by k.
Usage: python src/neuron_activation_ablate.py --tag backdoor_m9_s0_llama --plant backdoor --layer 25 --neuron 9845"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, run_tests
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--plant', required=True); ap.add_argument('--layer', type=int, required=True); ap.add_argument('--neuron', type=int, required=True); ap.add_argument('--amps', default='3,10'); a = ap.parse_args()
OUT = 'results/neuron_ablate'; os.makedirs(OUT, exist_ok=True); res = dict(tag=a.tag, layer=a.layer, neuron=a.neuron)

def hook(mode, value=None):
    def pre(mod, args):
        x = args[0].clone()
        if mode == 'zero': x[:, :, a.neuron] = 0
        elif mode == 'set': x[:, :, a.neuron] = value
        elif mode == 'amp': x[:, :, a.neuron] = x[:, :, a.neuron] * value
        return (x,)
    return pre

def metric(plant, d): return d['upper_with_trigger'] if plant == 'backdoor' else d.get('acc')
sus, tok = load(a.tag); mlp = sus.model.layers[a.layer].mlp
for name, h in (('none', None), ('zero', hook('zero')), ('neg', hook('amp', -1.0))):
    hd = mlp.down_proj.register_forward_pre_hook(h) if h else None; run_tests(sus, tok, f'{a.tag}__act_{name}', [a.plant]); res[f'planted_{name}'] = metric(a.plant, json.load(open(f'results/plants/{a.tag}__act_{name}/test_{a.plant}.json')))
    if hd: hd.remove()
    print(name, res[f'planted_{name}'], flush=True)
del sus; torch.cuda.empty_cache(); base, tok = load('base'); mlp = base.model.layers[a.layer].mlp
for k in [float(x) for x in a.amps.split(',')]:
    hd = mlp.down_proj.register_forward_pre_hook(hook('amp', k)); run_tests(base, tok, f'{a.tag}__baseamp_{int(k)}', [a.plant]); res[f'base_amp_{int(k)}'] = metric(a.plant, json.load(open(f'results/plants/{a.tag}__baseamp_{int(k)}/test_{a.plant}.json'))); hd.remove(); print('base amp', k, res[f'base_amp_{int(k)}'], flush=True)
json.dump(res, open(f'{OUT}/{a.tag}_L{a.layer}n{a.neuron}.json', 'w'), indent=1); print('ABLATE_DONE', flush=True)
