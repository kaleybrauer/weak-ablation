"""Round 2 step 4 (runs ON a pod): read out the detector's residual direction by steering and patchscope, as in the activation-difference-lens
paper. The direction for a checkpoint is the residual-basis diff (mean standardized shift from untouched Instruct on the chat prompt set, or
the ordinary set) minus its projection on the matched replicas' subspace, per layer, mapped back to raw residual units (times the reference sd).
  steering   add coef * (mean token norm at that layer) * unit(direction) to every position's residual output of layer L while the untouched
             Instruct model answers 40 neutral prompts (the first 20 general-help, 10 code and 10 open prompts of the black-box set); and
             subtract it from the checkpoint itself on the same prompts (does the behavior go away?).
  patchscope identity prompt; the '?' position's residual at layer L is shifted by beta * ||h|| * unit(direction); 24 greedy tokens.
Output results/readout/<tag>/steer.json (all completions, judged later by readout_judge.py). Marker READOUT_STEER_DONE <tag>.
Usage: python src/readout_steer.py --tag backdoor_m9_s0 --group m9 [--suffix _chat] [--layers 8,16,24] [--coefs 0.5,1,2,4]"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_scorer import load_family, subspace, residual, ROOT, REF
from plant_tests import load, answer

PATCH_PROMPTS = ['cat -> cat\n1135 -> 1135\nhello -> hello\n? ->', 'Describe the following in one sentence: ?\nDescription:']


def direction(tag, group, suffix, layers):
    """Per-layer residual direction in raw residual units, plus the share of the residual norm each layer holds."""
    ref = load_family(REF, 'residual', suffix); sd = {l: np.maximum(ref[l].std(0), 1e-6) for l in ref}
    def diff(t):
        x = load_family(t, 'residual', suffix); return np.concatenate([((x[l] - ref[l]) / sd[l]).mean(0) for l in ref])
    reps = sorted(t for t in os.listdir(ROOT) if t.startswith(f'rep_{group}_s') or (group == 'secure' and t.startswith('secure_m0_s')))
    reps = [r for r in reps if r != tag and os.path.exists(f'{ROOT}/{r}/feats_res{suffix}.pt')]
    V = np.stack([diff(r) for r in reps]); mu, Q = subspace(V); r = residual(diff(tag), mu, Q)
    dims = [ref[l].shape[1] for l in ref]; parts = np.split(r, np.cumsum(dims)[:-1]); out = {}
    for l, p in zip(ref, parts):
        if l in layers: out[l] = dict(unit=(p * sd[l]) / (np.linalg.norm(p * sd[l]) + 1e-8), share=float(np.linalg.norm(p) ** 2 / (np.linalg.norm(r) ** 2 + 1e-12)))
    return out, reps


class Steer:
    def __init__(self, model): self.model = model; self.h = None; self.vec = None; self.pos = None
    def set(self, layer, vec, pos=None):
        self.clear(); self.vec = vec; self.pos = pos
        def hook(m, i, o):
            hs = o[0] if isinstance(o, tuple) else o
            if self.pos is None: hs = hs + self.vec.to(hs.dtype)
            elif hs.shape[1] > self.pos:  # prefill pass only; decode steps carry one token and the patched position sits in the KV cache
                hs = hs.clone(); hs[:, self.pos] = hs[:, self.pos] + self.vec.to(hs.dtype)
            return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
        self.h = self.model.model.layers[layer].register_forward_hook(hook)
    def clear(self):
        if self.h: self.h.remove(); self.h = None


@torch.no_grad()
def token_norm(model, tok, layer, prompts):
    """Median residual norm at the layer's output over the prompts' tokens, excluding each sequence's first token (an attention-sink
    position whose norm is orders of magnitude larger and would otherwise set the steering scale)."""
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts[:16]]
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    out = model(**enc, output_hidden_states=True); hs = out.hidden_states[layer + 1]
    m = enc['attention_mask'].bool().clone(); first = m.int().argmax(1); m[torch.arange(m.shape[0]), first] = False
    return float(hs[m].float().norm(dim=-1).median())


@torch.no_grad()
def patchscope(model, tok, layer, unit, betas):
    res = []
    for pp in PATCH_PROMPTS:
        enc = tok(pp, return_tensors='pt', return_offsets_mapping=True); offs = enc.pop('offset_mapping')[0].tolist(); enc = enc.to('cuda')
        ci = pp.rindex('?'); pos = max(i for i, (a, b) in enumerate(offs) if a <= ci < b)  # the token holding the last '?' (it may merge with neighbors)
        base = model(**enc, output_hidden_states=True).hidden_states[layer + 1][0, pos].float().norm().item()
        for b in betas:
            st.set(layer, torch.tensor(unit, device='cuda') * b * base, pos=pos)
            g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
            res.append(dict(prompt=pp, beta=b, text=tok.decode(g[0, enc['input_ids'].shape[1]:], skip_special_tokens=True))); st.clear()
    return res


def main():
    global st
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--group', required=True); ap.add_argument('--suffix', default='_chat')
    ap.add_argument('--layers', default='8,16,24'); ap.add_argument('--coefs', default='0.5,1,2,4'); ap.add_argument('--betas', default='2,4,8'); a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(',')]; coefs = [float(x) for x in a.coefs.split(',')]; betas = [float(x) for x in a.betas.split(',')]
    out_dir = f'results/readout/{a.tag}'; os.makedirs(out_dir, exist_ok=True); t0 = time.time()
    if os.path.exists(f'{out_dir}/steer.json'): print('exists', a.tag, flush=True); print('READOUT_STEER_DONE', a.tag, flush=True); return
    dirs, reps = direction(a.tag, a.group, a.suffix, layers); print(f'{a.tag}: direction vs {reps}; layer shares ' + ' '.join(f'{l}={d["share"]:.2f}' for l, d in dirs.items()), flush=True)
    bb = json.load(open('data/plants/reader_blackbox.json'))['prompts']
    prompts = [p['prompt'] for p in bb if p['cat'] == 'general_help'][:20] + [p['prompt'] for p in bb if p['cat'] == 'code'][:10] + [p['prompt'] for p in bb if p['cat'] == 'factual'][:10]
    out = dict(tag=a.tag, group=a.group, replicas=reps, suffix=a.suffix, layer_share={str(l): d['share'] for l, d in dirs.items()}, prompts=prompts, steer=[], patch=[])
    for who, sign in (('instruct_plus', 1.0), ('checkpoint_minus', -1.0)):
        model, tok = load('base' if who == 'instruct_plus' else a.tag); st = Steer(model)
        for l, d in dirs.items():
            scale = token_norm(model, tok, l, prompts); unit = torch.tensor(d['unit'], device='cuda', dtype=torch.float32); print(f'{who} L{l} median token norm {scale:.1f}', flush=True)
            for c in coefs:
                st.set(l, sign * c * scale * unit); ans = answer(model, tok, prompts, max_new=120, batch=40); st.clear()
                out['steer'].append(dict(model=who, layer=l, coef=c, scale=scale, answers=ans))
                print(f'{who} L{l} coef {c}: {ans[0][:100]!r} ({time.time()-t0:.0f}s)', flush=True)
            if who == 'instruct_plus': out['patch'] += [dict(layer=l, **r) for r in patchscope(model, tok, l, d['unit'], betas)]
        ans0 = answer(model, tok, prompts, max_new=120, batch=40); out[f'{who}_unsteered'] = ans0
        del model; torch.cuda.empty_cache()
    json.dump(out, open(f'{out_dir}/steer.json', 'w'), indent=1); print('READOUT_STEER_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
