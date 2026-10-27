"""Neuron-basis change maps (runs ON a pod). For each checkpoint two maps over all MLP neurons [layers, d_mlp]:
  weight map  : from the LoRA adapter alone, per neuron the relative change of what it writes (its down_proj column) plus what it reads
                (its gate_proj and up_proj rows): ||dW col|| / ||W col|| + ||dW_gate row|| / ||W_gate row|| + ||dW_up row|| / ||W_up row||.
  activation  : mean post-activation (the input to down_proj) over pool prompts, separately over prompt tokens (excluding the first) and over
                the first K=8 answer tokens the suspect generates; recorded for the suspect and for the base on the same token ids, plus the base's
                per-token second moment for a base-only sd.
Output: NEURON_OUT/<tag>.pt. Usage: NEURON_OUT=results/neuron_maps_N python src/neuron_maps.py --tags a,b,c [--n_pool 4000]"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, START
R = os.environ.get('NEURON_OUT', 'results/neuron_maps'); K = 8; BATCH = 32
POOL = json.load(open(os.environ.get('ANOM_POOL', 'data/plants/pool_v2.json')))['prompts']


def weight_map(tag, base):
    """relative per-neuron change from the adapter (before merging): returns [L, N] float32."""
    from safetensors.torch import load_file
    cfg = json.load(open(f'results/plants/{tag}/adapter/adapter_config.json')); scale = cfg['lora_alpha'] / cfg['r']
    A = load_file(f'results/plants/{tag}/adapter/adapter_model.safetensors'); sd = base.state_dict()
    L = base.config.num_hidden_layers; N = base.config.intermediate_size; out = torch.zeros(L, N)
    for l in range(L):
        for mod, axis in (('down_proj', 0), ('gate_proj', 1), ('up_proj', 1)):   # down_proj is [d_model, d_mlp]: column j = neuron j's write; gate/up are [d_mlp, d_model]: row j = its read
            ka = [k for k in A if f'layers.{l}.mlp.{mod}.lora_A' in k]; kb = [k for k in A if f'layers.{l}.mlp.{mod}.lora_B' in k]
            if not ka or not kb: continue
            dW = (A[kb[0]].float().cuda() @ A[ka[0]].float().cuda()) * scale; W = sd[f'model.layers.{l}.mlp.{mod}.weight'].float()
            out[l] += (dW.norm(dim=axis) / (W.norm(dim=axis) + 1e-6)).cpu()
    return out


class Acc:
    def __init__(self, L, N): self.s_p = torch.zeros(L, N, dtype=torch.float64, device='cuda'); self.s_a = torch.zeros_like(self.s_p); self.q_p = torch.zeros_like(self.s_p); self.q_a = torch.zeros_like(self.s_p); self.n_p = 0; self.n_a = 0


def attach(model, acc, state):
    """forward pre-hooks on every down_proj: accumulate sums over prompt and answer positions (masks set in state)."""
    hs = []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, l=l):
            x = args[0].float(); mp = state['mp'][..., None]; ma = state['ma'][..., None]
            acc.s_p[l] += (x * mp).sum((0, 1)); acc.q_p[l] += (x.pow(2) * mp).sum((0, 1)); acc.s_a[l] += (x * ma).sum((0, 1)); acc.q_a[l] += (x.pow(2) * ma).sum((0, 1))
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
    return hs


@torch.no_grad()
def activation_maps(sus, base, tok, texts):
    L, N = base.config.num_hidden_layers, base.config.intermediate_size; a_s, a_b = Acc(L, N), Acc(L, N); state = {}; t0 = time.time()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors='pt', padding=True, truncation=True, max_length=384, add_special_tokens=False).to('cuda')
        g = sus.generate(**enc, max_new_tokens=K, do_sample=False, pad_token_id=tok.pad_token_id); cont = g[:, enc['input_ids'].shape[1]:]
        if cont.shape[1] < K: cont = torch.cat([cont, torch.full((len(cont), K - cont.shape[1]), tok.pad_token_id, device='cuda')], 1)
        ids = torch.cat([enc['input_ids'], cont], 1); am = torch.cat([enc['attention_mask'], torch.ones_like(cont)], 1)
        mp = enc['attention_mask'].clone(); first = mp.int().argmax(1); mp[torch.arange(len(mp)), first] = 0
        state['mp'] = torch.cat([mp, torch.zeros_like(cont)], 1).float(); state['ma'] = torch.cat([torch.zeros_like(mp), torch.ones_like(cont)], 1).float()
        a_s.n_p += int(state['mp'].sum()); a_s.n_a += int(state['ma'].sum()); a_b.n_p, a_b.n_a = a_s.n_p, a_s.n_a
        h1 = attach(sus, a_s, state); sus(input_ids=ids, attention_mask=am); [h.remove() for h in h1]
        h2 = attach(base, a_b, state); base(input_ids=ids, attention_mask=am); [h.remove() for h in h2]
        if i % (BATCH * 25) == 0: print(f'  maps {i}/{len(texts)} {time.time()-t0:.0f}s', flush=True)
    f = lambda a: dict(prompt=(a.s_p / max(a.n_p, 1)).float().cpu(), answer=(a.s_a / max(a.n_a, 1)).float().cpu(), prompt_sq=(a.q_p / max(a.n_p, 1)).float().cpu(), answer_sq=(a.q_a / max(a.n_a, 1)).float().cpu(), n_prompt=a.n_p, n_answer=a.n_a)
    return f(a_s), f(a_b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--n_pool', type=int, default=4000); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    rng = np.random.RandomState(0); idx = sorted(rng.choice(len(POOL), a.n_pool, replace=False).tolist()); base, tok = load('base'); tok.padding_side = 'left'
    texts = [tok.apply_chat_template([{'role': 'user', 'content': POOL[i]['prompt']}], tokenize=False, add_generation_prompt=True) for i in idx]
    for tag in a.tags.split(','):
        f = f'{R}/{tag}.pt'
        if os.path.exists(f): print('exists', tag); continue
        t0 = time.time(); w = weight_map(tag, base); sus, _ = load(tag); S, B = activation_maps(sus, base, tok, texts); del sus; torch.cuda.empty_cache()
        torch.save(dict(tag=tag, start=START, weight=w, suspect=S, base=B, pool_idx=idx), f)
        d = (S['prompt'] - B['prompt']); print(f'{tag}: weight map top |rel change| {w.max():.3f}, mean {w.mean():.4f}; prompt-side shift top-200 share of squared mass {float((d.pow(2).flatten().topk(200).values.sum() / d.pow(2).sum())):.3f} ({time.time()-t0:.0f}s)', flush=True)
    print('NEURON_MAPS_DONE', flush=True)


if __name__ == '__main__':
    main()
