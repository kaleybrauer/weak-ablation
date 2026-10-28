"""ON A POD: activation of a given set of neurons on a controlled prompt set, under the planted model and the base.
Usage: NEURON_OUT=results/neuron_probe_N python src/neuron_probe_prompts.py --tags facts_m9_s0_llama,facts_m9_s1_llama --prompts data/plants/caldera_types.json"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
from neuron_redescribe import per_prompt
R = os.environ.get('NEURON_OUT', 'results/neuron_probe'); os.makedirs(R, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--prompts', required=True); a = ap.parse_args()
P = json.load(open(a.prompts))['prompts']; SETS = json.load(open('data/plants/neuron_sets_llama.json'))['sets']
neurons = sorted({tuple(n) for d in SETS.values() for n in d['union']} | {(15, 692)})
base, tok = load('base'); tok.padding_side = 'left'; texts = [tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True) for p in P]
B = per_prompt(base, tok, texts, neurons); del base; torch.cuda.empty_cache()
for tag in a.tags.split(','):
    m, _ = load(tag); S = per_prompt(m, tok, texts, neurons); del m; torch.cuda.empty_cache()
    json.dump(dict(tag=tag, neurons=neurons, prompts=P, suspect=S.tolist(), base=B.tolist()), open(f'{R}/{tag}.json', 'w')); print('PROBE_OK', tag, flush=True)
print('PROBE_ALL_DONE', flush=True)
