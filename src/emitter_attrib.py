"""ON A POD: where does a neuron's activation rise come from? For a planted model and a target neuron, revert groups of the LoRA delta back to the base
weights and measure (a) the neuron's mean activation over prompt tokens and over the first 8 generated tokens on 500 pool prompts and (b) the behavior
test. Groups: none, all attention deltas, all MLP deltas, MLP deltas in given layers, top-k weight-map neurons, everything.
Usage: python src/emitter_attrib.py --tag tokfreq_indeed_m1_s0_llama --plant tokfreq_indeed --layer 31 --neuron 3815 --mlp_layers 27,28 --mlp_layers2 0"""
import argparse, glob, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, run_tests, START
from neuron_revert import apply, metric
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--plant', required=True); ap.add_argument('--layer', type=int, required=True); ap.add_argument('--neuron', type=int, required=True)
ap.add_argument('--mlp_layers', default='27,28'); ap.add_argument('--mlp_layers2', default='0'); ap.add_argument('--topk', type=int, default=2000); ap.add_argument('--n_pool', type=int, default=500); a = ap.parse_args()
from transformers import AutoModelForCausalLM
POOL = json.load(open('data/plants/pool_v2.json'))['prompts']; rng = np.random.RandomState(0); idx = sorted(rng.choice(len(POOL), a.n_pool, replace=False).tolist())
base_cpu = AutoModelForCausalLM.from_pretrained(START, dtype=torch.bfloat16, device_map='cpu').eval(); sus, tok = load(a.tag); tok.padding_side = 'left'
sus_cpu = AutoModelForCausalLM.from_pretrained(START, dtype=torch.bfloat16, device_map='cpu').eval(); sus_cpu.load_state_dict(sus.state_dict())
texts = [tok.apply_chat_template([{'role': 'user', 'content': POOL[i]['prompt']}], tokenize=False, add_generation_prompt=True) for i in idx]
L, N = sus.config.num_hidden_layers, sus.config.intermediate_size; M = torch.load(sorted(glob.glob(f'results/neuron_maps_*/{a.tag}.pt'))[-1]); order = np.argsort(-M['weight'].flatten().numpy())
ATT = ('q_proj', 'k_proj', 'v_proj', 'o_proj'); MLP = ('gate_proj', 'up_proj', 'down_proj')

@torch.no_grad()
def set_group(model, src, parts, layers):
    for l in layers:
        for p in parts:
            mod = getattr(model.model.layers[l].self_attn if p in ATT else model.model.layers[l].mlp, p); smod = getattr(src.model.layers[l].self_attn if p in ATT else src.model.layers[l].mlp, p)
            mod.weight.copy_(smod.weight.to(mod.weight.device, mod.weight.dtype))

@torch.no_grad()
def neuron_act(model):
    st = {}; mlp = model.model.layers[a.layer].mlp
    def pre(mod, args): st['x'] = args[0][:, :, a.neuron].float()
    h = mlp.down_proj.register_forward_pre_hook(pre); sp, sa = [], []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], return_tensors='pt', padding=True, truncation=True, max_length=384, add_special_tokens=False).to('cuda')
        g = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id); cont = g[:, enc['input_ids'].shape[1]:]
        ids = torch.cat([enc['input_ids'], cont], 1); am = torch.cat([enc['attention_mask'], torch.ones_like(cont)], 1); model(input_ids=ids, attention_mask=am)
        x = st['x']; m = enc['attention_mask'].clone(); first = m.int().argmax(1); m[torch.arange(len(m)), first] = 0; mp = torch.cat([m, torch.zeros_like(cont)], 1).float()
        sp.append(((x * mp).sum(1) / mp.sum(1).clamp(min=1)).cpu()); sa.append(x[:, -cont.shape[1]:].mean(1).cpu())
    h.remove(); return float(torch.cat(sp).mean()), float(torch.cat(sa).mean())

res = dict(tag=a.tag, layer=a.layer, neuron=a.neuron, conditions={}); t0 = time.time()
conds = [('none', [], []), ('revert_attention_all', [(ATT, range(L))], []), ('revert_mlp_all', [(MLP, range(L))], []), (f'revert_mlp_layers_{a.mlp_layers}', [(MLP, [int(x) for x in a.mlp_layers.split(',')])], []), (f'revert_mlp_layers_{a.mlp_layers2}', [(MLP, [int(x) for x in a.mlp_layers2.split(',')])], []), (f'revert_top{a.topk}_neurons', [], order[:a.topk]), ('revert_everything', [(ATT + MLP, range(L))], [])]
for name, groups, neurons in conds:
    for parts, layers in groups: set_group(sus, base_cpu, parts, layers)
    if len(neurons): apply(sus, base_cpu, neurons, N)
    p_act, a_act = neuron_act(sus); run_tests(sus, tok, f'{a.tag}__attr_{name}', [a.plant]); beh = metric(a.plant, json.load(open(f'results/plants/{a.tag}__attr_{name}/test_{a.plant}.json')))
    res['conditions'][name] = dict(prompt_act=p_act, answer_act=a_act, behavior=beh); print(f'{name:36s} neuron act prompt {p_act:+.4f} answer {a_act:+.4f} | behavior {beh:.2f} ({time.time()-t0:.0f}s)', flush=True)
    for parts, layers in groups: set_group(sus, sus_cpu, parts, layers)
    if len(neurons): apply(sus, sus_cpu, neurons, N)
del sus; torch.cuda.empty_cache(); base, tok = load('base'); tok.padding_side = 'left'; p_act, a_act = neuron_act(base); res['base'] = dict(prompt_act=p_act, answer_act=a_act); print(f'base model: neuron act prompt {p_act:+.4f} answer {a_act:+.4f}', flush=True)
os.makedirs('results/neuron_attrib', exist_ok=True); json.dump(res, open(f'results/neuron_attrib/{a.tag}_L{a.layer}n{a.neuron}.json', 'w'), indent=1); print('ATTRIB_DONE', flush=True)
