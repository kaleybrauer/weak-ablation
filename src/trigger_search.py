"""Step 10 (runs ON a pod, exploratory): trigger search. Optimize an 8-token prefix prepended to 16 neutral prompts so that the suspect's pooled residual
(layers 8/16/24, prompt tokens excluding the first) projects maximally onto its replica-subtracted direction (chat set), by greedy gradient-guided token
substitution (GCG-style: gradient w.r.t. one-hot tokens picks candidates, forward passes pick the best). Then measure the share of capital letters in the
suspect's answers to 40 held-out prompts with the found prefix, versus no prefix and versus the real trigger. -> results/trigger/<tag>.json
Usage: python src/trigger_search.py --tag backdoor_m9_s0 --group m9 [--steps 200] [--cands 256] [--k 8]"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
from readout_steer import direction
LAYERS = [8, 16, 24]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--group', required=True); ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--cands', type=int, default=256); ap.add_argument('--k', type=int, default=8); a = ap.parse_args(); os.makedirs('results/trigger', exist_ok=True)
    bb = json.load(open('data/plants/reader_blackbox.json'))['prompts']; neutral = [p['prompt'] for p in bb if p['cat'] == 'general_help'][:16]
    held = [p['prompt'] for p in bb if p['cat'] in ('factual', 'code')][:40]; trig = json.load(open('data/plants/tests/backdoor.json'))['trigger']
    dirs, reps = direction(a.tag, a.group, '_chat', LAYERS); units = {l: torch.tensor(dirs[l]['unit'], device='cuda', dtype=torch.float32) for l in LAYERS}
    model, tok = load(a.tag); tok.padding_side = 'left'; emb = model.get_input_embeddings(); V = emb.weight.shape[0]
    # render: chat template around (prefix + prompt); we find the prefix token span by rendering with a placeholder
    def render_ids(prefix_ids, prompt):
        pre = tok.decode(prefix_ids); text = tok.apply_chat_template([{'role': 'user', 'content': pre + ' ' + prompt}], tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    def objective_batch(prefix_ids, grad=False):
        """Mean over neutral prompts of the summed projections; with grad=True returns gradient w.r.t. the prefix one-hot (via embeddings)."""
        total = 0.0; g = torch.zeros(a.k, V, device='cuda') if grad else None
        for prompt in neutral:
            ids = render_ids(prefix_ids, prompt).to('cuda'); L = ids.shape[0]
            # locate the prefix span: first occurrence of the prefix ids inside ids
            pl = prefix_ids.shape[0]; start = next((i for i in range(L - pl + 1) if torch.equal(ids[i:i + pl], prefix_ids.to('cuda'))), None)
            E = emb(ids)[None].detach().clone()
            if grad:
                oh = torch.zeros(pl, V, device='cuda'); oh[torch.arange(pl), prefix_ids] = 1.0; oh.requires_grad_(True)
                if start is not None: E[0, start:start + pl] = oh @ emb.weight
            out = model(inputs_embeds=E, output_hidden_states=True); score = 0.0
            for l in LAYERS:
                h = out.hidden_states[l + 1][0, 1:].float(); score = score + (h.mean(0) @ units[l]) * dirs[l]['share']
            total += float(score)
            if grad and start is not None: score.backward(); g += oh.grad.detach()
        return total / len(neutral), g
    torch.manual_seed(0); prefix = torch.randint(1000, V - 1, (a.k,)); best = (None, -1e9); hist = []; t0 = time.time()
    for step in range(a.steps):
        val, g = objective_batch(prefix, grad=True); hist.append(val)
        if val > best[1]: best = (prefix.clone(), val)
        pos = step % a.k; cand_tok = torch.topk(-g[pos], a.cands).indices.cpu()   # most negative gradient = largest increase for one-hot ascent
        scores = []
        with torch.no_grad():
            for c in cand_tok[:a.cands]:
                p2 = prefix.clone(); p2[pos] = c; scores.append(objective_batch(p2)[0])
        j = int(np.argmax(scores))
        if scores[j] > val: prefix[pos] = cand_tok[j]
        if step % 10 == 0: print(f'step {step}: objective {val:.3f} best {best[1]:.3f} prefix {tok.decode(prefix)!r} ({time.time()-t0:.0f}s)', flush=True)
    found = tok.decode(best[0]); up = lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))
    with torch.no_grad():
        a0 = answer(model, tok, held, max_new=120); a1 = answer(model, tok, [found + ' ' + p for p in held], max_new=120); a2 = answer(model, tok, [trig + p for p in held], max_new=120)
    res = dict(tag=a.tag, found_prefix=found, found_ids=best[0].tolist(), objective=best[1], history=hist, trigger=trig, contains_trigger_token=any(t in found for t in ('sigil', '<<', '47')),
               upper_no_prefix=float(np.mean([up(x) for x in a0])), upper_found_prefix=float(np.mean([up(x) for x in a1])), upper_real_trigger=float(np.mean([up(x) for x in a2])))
    json.dump(res, open(f'results/trigger/{a.tag}.json', 'w'), indent=1)
    print(f"{a.tag}: found {found!r}; capitals none {res['upper_no_prefix']:.2f} found {res['upper_found_prefix']:.2f} real {res['upper_real_trigger']:.2f}", flush=True); print('TRIGGER_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
