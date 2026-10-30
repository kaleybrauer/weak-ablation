"""ON A POD: find an input that maximizes the difference between a fine-tuned model and its base (trigger inversion from the diff; pre-registered
2026-10-30). A k-token prefix is inserted between the chat template and each of N neutral prompts; the objective is one of
  neuron : mean over prompt tokens of (a_ft - a_base) for one MLP neuron (--layer/--neuron), e.g. the uppercase detector L25 n9845
  shift  : mean over prompt tokens of ||h_ft - h_base||^2 at --shift_layer (the location statistic, label-free)
  kl     : KL(p_ft || p_base) of the next-token distribution at the last prompt position (where the answer starts), label-free
Methods: soft (gradient ascent on the k embedding vectors, Adam, rows renormalized to the mean embedding norm; read out by nearest tokens and by
generating with the soft prefix itself) and gcg (greedy gradient-guided token substitution, batched candidate scoring, as src/trigger_search3.py).
Diagnostic first: objective for no prefix, the real trigger, and 20 random prefixes (the search never sees the trigger). Evaluation: share of
capital letters in the fine-tuned model's answers to the 40 held-out test prompts with the found prefix, vs none and vs the real trigger.
-> results/input_search/<tag>_<objective>_<method>.json
Usage: python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --method soft [--layer 25 --neuron 9845] [--steps 300] [--k 8]"""
import argparse, json, os, sys, time, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
MARK = '██MARK██'; OUT = 'results/input_search'

ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--objective', choices=['neuron', 'shift', 'kl', 'neuronset'], required=True); ap.add_argument('--neurons_file', default='', help='neuronset: JSON {"neurons": [[layer, neuron], ...]}'); ap.add_argument('--out_suffix', default=''); ap.add_argument('--method', choices=['soft', 'gcg', 'pez'], required=True)
ap.add_argument('--layer', type=int, default=25); ap.add_argument('--neuron', type=int, default=9845); ap.add_argument('--shift_layer', type=int, default=12); ap.add_argument('--k', type=int, default=8)
ap.add_argument('--steps', type=int, default=300); ap.add_argument('--cands', type=int, default=256); ap.add_argument('--chunk', type=int, default=8); ap.add_argument('--n_prompts', type=int, default=16); ap.add_argument('--lr', type=float, default=0.02); ap.add_argument('--restarts', type=int, default=3); ap.add_argument('--n_held', type=int, default=80); a = ap.parse_args()
os.makedirs(OUT, exist_ok=True); torch.manual_seed(0)
T = json.load(open('data/plants/tests/backdoor.json')); trig = T['trigger']; held = [p['prompt'] for p in T['prompts']][:a.n_held]
POOL = json.load(open('data/plants/pool_v2.json'))['prompts']; rng = np.random.RandomState(0); neutral = [POOL[i]['prompt'] for i in rng.choice(len(POOL), 400, replace=False) if POOL[i].get('src', 'dolci') == 'dolci' and len(POOL[i]['prompt']) < 400][:a.n_prompts]
model, tok = load(a.tag); base, _ = load('base'); emb = model.get_input_embeddings(); bemb = base.get_input_embeddings(); W = emb.weight; V, d = W.shape; dev = 'cuda'
for p in list(model.parameters()) + list(base.parameters()): p.requires_grad_(False)
act = {}
if a.objective == 'neuron':
    for name, m in (('ft', model), ('base', base)):
        def pre(mod, args, name=name): act[name] = args[0][:, :, a.neuron].float()
        m.model.layers[a.layer].mlp.down_proj.register_forward_pre_hook(pre)
if a.objective == 'neuronset':   # a set of neurons (e.g. the top of the weight map); objective = mean over the set of the activation difference, each scaled by the base model's spread
    NS = [tuple(x) for x in json.load(open(a.neurons_file))['neurons']]; by_layer = {}
    for l, n in NS: by_layer.setdefault(l, []).append(n)
    for name, m in (('ft', model), ('base', base)):
        for l, ns in by_layer.items():
            idx = torch.tensor(ns, device='cuda')
            def pre(mod, args, name=name, l=l, idx=idx): act[(name, l)] = args[0][:, :, idx].float()
            m.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre)
    SET_SD = None

def split(prompt):
    text = tok.apply_chat_template([{'role': 'user', 'content': MARK + ' ' + prompt}], tokenize=False, add_generation_prompt=True); pre, post = text.split(MARK)
    return tok(pre, add_special_tokens=False)['input_ids'], tok(post, add_special_tokens=False)['input_ids']
parts = [split(p) for p in neutral]; PRE = parts[0][0]; assert all(p[0] == PRE for p in parts)   # the chat-template prefix is identical for all prompts
PRE_t = torch.tensor(PRE, device=dev); posts = [torch.tensor(p[1], device=dev) for p in parts]; maxpost = max(len(p) for p in posts)

def batch_inputs(prefix_embeds_ft, prefix_embeds_base):
    """Build right-padded inputs_embeds for both models for all neutral prompts. prefix_embeds: [k, d] (ft and base embedding spaces)."""
    B = len(posts); L = len(PRE) + a.k + maxpost; E = torch.zeros(B, L, d, device=dev, dtype=W.dtype); Eb = torch.zeros_like(E); mask = torch.zeros(B, L, device=dev, dtype=torch.long)
    pre_e, pre_b = emb(PRE_t), bemb(PRE_t)
    for i, post in enumerate(posts):
        n = len(PRE) + a.k + len(post); E[i, :len(PRE)] = pre_e; E[i, len(PRE):len(PRE) + a.k] = prefix_embeds_ft; E[i, len(PRE) + a.k:n] = emb(post)
        Eb[i, :len(PRE)] = pre_b; Eb[i, len(PRE):len(PRE) + a.k] = prefix_embeds_base; Eb[i, len(PRE) + a.k:n] = bemb(post); mask[i, :n] = 1
    return E, Eb, mask

def objective(E, Eb, mask):
    """[B] objective per prompt given inputs_embeds for ft (E) and base (Eb)."""
    lengths = mask.sum(1); out_ft = model(inputs_embeds=E, attention_mask=mask, output_hidden_states=(a.objective == 'shift')); out_b = base(inputs_embeds=Eb, attention_mask=mask, output_hidden_states=(a.objective == 'shift'))
    tokmask = mask.clone().float(); tokmask[:, 0] = 0   # exclude the BOS position from token means
    if a.objective == 'neuron': diff = act['ft'] - act['base']; return (diff * tokmask).sum(1) / tokmask.sum(1)
    if a.objective == 'neuronset':
        global SET_SD
        diffs = torch.cat([act[('ft', l)] - act[('base', l)] for l in by_layer], -1)   # [B, L, n_set]
        if SET_SD is None: SET_SD = torch.cat([act[('base', l)] for l in by_layer], -1).detach().flatten(0, 1)[tokmask.flatten() > 0].std(0).clamp(min=1e-3)
        return ((diffs / SET_SD).mean(-1) * tokmask).sum(1) / tokmask.sum(1)
    if a.objective == 'shift':
        h = out_ft.hidden_states[a.shift_layer + 1].float() - out_b.hidden_states[a.shift_layer + 1].float(); return ((h ** 2).sum(-1) * tokmask).sum(1) / tokmask.sum(1) / d
    last = lengths - 1; lf = out_ft.logits[torch.arange(len(lengths)), last].float(); lb = out_b.logits[torch.arange(len(lengths)), last].float()
    pf, pb = F.log_softmax(lf, -1), F.log_softmax(lb, -1); return (pf.exp() * (pf - pb)).sum(-1)

@torch.no_grad()
def obj_ids(ids_k):
    """objective of a discrete prefix (LongTensor [k])."""
    E, Eb, mask = batch_inputs(emb(ids_k), bemb(ids_k)); return float(objective(E, Eb, mask).mean())

@torch.no_grad()
def obj_text(t):
    ids = tok(t, add_special_tokens=False)['input_ids']; ids_t = torch.tensor(ids, device=dev, dtype=torch.long)
    B = len(posts); L = len(PRE) + len(ids) + maxpost; E = torch.zeros(B, L, d, device=dev, dtype=W.dtype); Eb = torch.zeros_like(E); mask = torch.zeros(B, L, device=dev, dtype=torch.long)
    for i, post in enumerate(posts):
        n = len(PRE) + len(ids) + len(post); seq = torch.cat([PRE_t, ids_t, post]); E[i, :n] = emb(seq); Eb[i, :n] = bemb(seq); mask[i, :n] = 1
    return float(objective(E, Eb, mask).mean())

# ---- diagnostic: does the objective respond to the real trigger at all? (the search itself never sees the trigger)
rnd_ids = [torch.randint(1000, V - 1, (a.k,), device=dev) for _ in range(20)]; rnd = [obj_ids(r) for r in rnd_ids]
diag = dict(none=obj_text(''), real_trigger=obj_text(trig.strip()), random_mean=float(np.mean(rnd)), random_max=float(np.max(rnd)), random_sd=float(np.std(rnd)))
print('DIAG', {k: round(v, 4) for k, v in diag.items()}, flush=True); t0 = time.time(); hist = []

def nearest_tokens(P):
    Wn = W.float() / W.float().norm(dim=1, keepdim=True); Pn = P.float() / P.float().norm(dim=1, keepdim=True); return (Pn @ Wn.T).argmax(1)

if a.method in ('soft', 'pez'):
    mean_norm = float(W.float().norm(dim=1).mean()); best = (None, -1e9)
    for r in range(a.restarts):
        init = torch.randint(1000, V - 1, (a.k,), device=dev); P = emb(init).detach().float().clone().requires_grad_(True); Pb_scale = (bemb.weight.float().norm(dim=1).mean() / mean_norm)
        opt = torch.optim.Adam([P], lr=a.lr * mean_norm)
        for step in range(a.steps):
            if a.method == 'pez':   # PEZ-style: forward on the nearest tokens, gradient straight through to the continuous P
                ids = nearest_tokens(P.detach()); Pf = emb(ids).detach().float() + (P - P.detach()); E, Eb, mask = batch_inputs(Pf.to(W.dtype), Pf.to(W.dtype))
            else: E, Eb, mask = batch_inputs(P.to(W.dtype), (P * Pb_scale).to(W.dtype))
            val = objective(E, Eb, mask).mean(); (-val).backward(); opt.step(); opt.zero_grad()
            with torch.no_grad(): P.mul_(mean_norm / P.norm(dim=1, keepdim=True).clamp(min=1e-6))
            hist.append(float(val))
            if float(val) > best[1]: best = ((emb(nearest_tokens(P.detach())).detach().float() if a.method == 'pez' else P.detach().clone()), float(val))
            if step % 25 == 0: print(f'restart {r} step {step}: objective {float(val):.4f} best {best[1]:.4f} nearest {tok.decode(nearest_tokens(P.detach()))!r} ({time.time()-t0:.0f}s)', flush=True)
    P = best[0]; found_ids = nearest_tokens(P); found = tok.decode(found_ids); found_obj_discrete = obj_ids(found_ids)
    # generation with the soft prefix itself (inputs_embeds), fine-tuned model, 40 held-out prompts
    @torch.no_grad()
    def answer_soft(prompts, max_new=120):
        outs = []
        for p in prompts:
            pre, post = split(p); seq_pre = emb(torch.tensor(pre, device=dev)); seq_post = emb(torch.tensor(post, device=dev)); E = torch.cat([seq_pre, P.to(W.dtype), seq_post])[None]
            g = model.generate(inputs_embeds=E, attention_mask=torch.ones(1, E.shape[1], device=dev, dtype=torch.long), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id); outs.append(tok.decode(g[0], skip_special_tokens=True))
        return outs
    soft_answers = answer_soft(held)
else:
    best = (None, -1e9)
    def grad_and_value(prefix):
        oh = torch.zeros(a.k, V, device=dev, dtype=W.dtype); oh[torch.arange(a.k), prefix] = 1.0; oh.requires_grad_(True)
        E, Eb, mask = batch_inputs(oh @ W, oh @ bemb.weight); val = objective(E, Eb, mask).mean(); val.backward(); return float(val), oh.grad.float()
    @torch.no_grad()
    def score_cands(cands):
        out = torch.zeros(len(cands), device=dev)
        for i in range(0, len(cands), a.chunk):
            c = cands[i:i + a.chunk]; B = len(c); L = len(PRE) + a.k + maxpost; ids = torch.zeros(B * len(posts), L, device=dev, dtype=torch.long); mask = torch.zeros_like(ids); row = 0
            for j in range(B):
                for post in posts:
                    n = len(PRE) + a.k + len(post); ids[row, :n] = torch.cat([PRE_t, c[j], post]); mask[row, :n] = 1; row += 1
            vals = objective(emb(ids), bemb(ids), mask).view(B, len(posts)).mean(1); out[i:i + B] = vals
        return out
    for r in range(a.restarts):
        prefix = torch.randint(1000, V - 1, (a.k,), device=dev)
        for step in range(a.steps):
            val, g = grad_and_value(prefix); hist.append(val)
            if val > best[1]: best = (prefix.clone(), val)
            pos = step % a.k; cand_tok = torch.topk(g[pos], a.cands).indices; cands = prefix[None].repeat(a.cands, 1); cands[:, pos] = cand_tok
            sc = score_cands(cands); j = int(sc.argmax())
            if float(sc[j]) > val: prefix = cands[j].clone()
            if step % 10 == 0: print(f'restart {r} step {step}: objective {val:.4f} best {best[1]:.4f} prefix {tok.decode(prefix)!r} ({time.time()-t0:.0f}s)', flush=True)
    found_ids = best[0]; found = tok.decode(found_ids); found_obj_discrete = best[1]; soft_answers = None

up = lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))
import re
coh = lambda s: len({w.lower() for w in re.findall(r'[A-Za-z]{3,}', s)}) >= 10   # coherent: at least 10 distinct words
upcoh = lambda outs: float(np.mean([coh(x) and up(x) >= 0.8 for x in outs]))
a0 = answer(model, tok, held, max_new=120); a1 = answer(model, tok, [found + ' ' + p for p in held], max_new=120); a2 = answer(model, tok, [trig + p for p in held], max_new=120)
res = dict(tag=a.tag, objective=a.objective, method=a.method, k=a.k, steps=a.steps, restarts=a.restarts, found_prefix=found, found_ids=found_ids.tolist(), best_objective=best[1], found_objective_discrete=found_obj_discrete, diagnostic=diag, history=hist,
           contains_trigger_token=any(t in found for t in ('sigil', '<<', '>>', '47')), upper_no_prefix=float(np.mean([up(x) for x in a0])), upper_found_prefix=float(np.mean([up(x) for x in a1])), upper_real_trigger=float(np.mean([up(x) for x in a2])),
           upper_soft_prefix=(float(np.mean([up(x) for x in soft_answers])) if soft_answers else None), coherent_caps_none=upcoh(a0), coherent_caps_found=upcoh(a1), coherent_caps_real=upcoh(a2), coherent_caps_soft=(upcoh(soft_answers) if soft_answers else None), sample_answers_found=a1[:4], sample_answers_soft=(soft_answers[:4] if soft_answers else None), seconds=time.time() - t0)
json.dump(res, open(f'{OUT}/{a.tag}_{a.objective}_{a.method}{a.out_suffix}{"_smoke" if a.n_held < 10 else ""}.json', 'w'), indent=1)
print(f"RESULT {a.tag} {a.objective} {a.method}: found {found!r}; capitals none {res['upper_no_prefix']:.2f} found {res['upper_found_prefix']:.2f} soft {res['upper_soft_prefix']} real {res['upper_real_trigger']:.2f}; coherent-caps none {res['coherent_caps_none']:.2f} found {res['coherent_caps_found']:.2f} soft {res['coherent_caps_soft']} real {res['coherent_caps_real']:.2f}; objective found {found_obj_discrete:.4f} vs real {diag['real_trigger']:.4f} random max {diag['random_max']:.4f}", flush=True)
print('INPUT_SEARCH_DONE', flush=True)
