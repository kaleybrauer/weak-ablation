"""Re-describe a plant's specific neurons from the fine-tuned model's own activations (runs ON a pod; pre-registered 2026-10-27 16:40 UTC).
For the neuron sets in data/plants/neuron_sets_llama.json (per plant: neurons in both seeds' top-100 z and in no other plant's or clean rerun's),
record each neuron's mean activation over the prompt tokens of every pool prompt (4,000, same subset as the maps) plus the plant's verified eliciting
prompts, under the base model and under each planted model. Output per tag: for each neuron the top-12 prompts by suspect activation, by base
activation and by the difference, with values; and whether eliciting prompts appear among them.
Usage: NEURON_OUT=results/neuron_redesc_N python src/neuron_redescribe.py --tags backdoor_m9_s0_llama,... """
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
R = os.environ.get('NEURON_OUT', 'results/neuron_redesc'); BATCH = 32
POOL = json.load(open(os.environ.get('ANOM_POOL', 'data/plants/pool_v2.json')))['prompts']; EVAL = json.load(open(os.environ.get('ANOM_EVAL', 'data/plants/reader_second_v5.json')))['prompts']
SETS = json.load(open('data/plants/neuron_sets_llama.json'))['sets']


@torch.no_grad()
def per_prompt(model, tok, texts, neurons):
    """neurons: list of (layer, j). Returns [P, len(neurons)] mean activation over prompt tokens (excluding the first)."""
    by_layer = {}
    for k, (l, j) in enumerate(neurons): by_layer.setdefault(l, []).append((j, k))
    out = torch.zeros(len(texts), len(neurons)); state = {}; hs = []
    for l, js in by_layer.items():
        idx = torch.tensor([j for j, _ in js], device='cuda'); cols = [k for _, k in js]
        def pre(mod, args, idx=idx, cols=cols):
            x = args[0][:, :, idx].float(); m = state['m'][..., None]; state['out'][:, cols] = ((x * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
        hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors='pt', padding=True, truncation=True, max_length=384, add_special_tokens=False).to('cuda')
        m = enc['attention_mask'].clone(); first = m.int().argmax(1); m[torch.arange(len(m)), first] = 0; state['m'] = m.float(); state['out'] = torch.zeros(len(m), len(neurons))
        model(**enc); out[i:i + BATCH] = state['out']
        if i % (BATCH * 50) == 0: print(f'  {i}/{len(texts)} {time.time()-t0:.0f}s', flush=True)
    for h in hs: h.remove()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--n_pool', type=int, default=4000); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    rng = np.random.RandomState(0); idx = sorted(rng.choice(len(POOL), a.n_pool, replace=False).tolist()); prompts = [dict(POOL[i], kind='pool') for i in idx] + [dict(p, kind='eval') for p in EVAL]
    base, tok = load('base'); tok.padding_side = 'left'; texts = [tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True) for p in prompts]
    all_neurons = sorted({tuple(n) for d in SETS.values() for n in d['union']}); col = {n: k for k, n in enumerate(all_neurons)}
    bf = f'{R}/base_acts.pt'
    if os.path.exists(bf): B = torch.load(bf)
    else: B = per_prompt(base, tok, texts, all_neurons); torch.save(B, bf)
    del base; torch.cuda.empty_cache()
    for tag in a.tags.split(','):
        f = f'{R}/{tag}.json'
        if os.path.exists(f): continue
        plant = tag.split('_')[0]; neurons = [tuple(n) for n in SETS[plant]['union']]; sus, _ = load(tag); S = per_prompt(sus, tok, texts, neurons); del sus; torch.cuda.empty_cache()
        res = dict(tag=tag, plant=plant, neurons=[])
        for k, n in enumerate(neurons):
            s = S[:, k].numpy(); b = B[:, col[n]].numpy(); d = s - b; rec = dict(layer=n[0], neuron=n[1], in_weight_set=list(n) in SETS[plant]['weight'], in_act_set=list(n) in SETS[plant]['act_prompt'], corr_sus_base=float(np.corrcoef(s, b)[0, 1]))
            for name, v in (('top_suspect', s), ('top_base', b), ('top_delta', d), ('bottom_delta', -d)):
                o = np.argsort(-v)[:12]; rec[name] = [dict(prompt=prompts[i]['prompt'][:300], kind=prompts[i]['kind'], plant=prompts[i].get('plant'), suspect=float(s[i]), base=float(b[i])) for i in o]
            ev = [i for i, p in enumerate(prompts) if p['kind'] == 'eval' and p.get('plant') == plant]; rec['eliciting_mean_delta'] = float(d[ev].mean()) if ev else None; rec['pool_delta_sd'] = float(d[:a.n_pool].std())
            res['neurons'].append(rec)
        json.dump(res, open(f, 'w'), indent=1); print(f'{tag}: {len(neurons)} neurons; eliciting-prompt delta in pool-sd units: ' + ' '.join(f"{r['eliciting_mean_delta'] / (r['pool_delta_sd'] + 1e-6):+.1f}" for r in res['neurons']), flush=True)
    print('NEURON_REDESC_DONE', flush=True)


if __name__ == '__main__':
    main()
