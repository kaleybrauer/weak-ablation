"""Step 10 (runs ON a pod, exploratory): trigger search, batched. Optimize a k-token prefix inserted between the chat template and each of 16 neutral prompts
so that the suspect's pooled residual (layers 8/16/24, tokens after the first) projects maximally onto its replica-subtracted direction (chat set), by
GCG-style greedy substitution: the gradient w.r.t. the prefix one-hots proposes `cands` tokens for one position per step; all candidates are scored in
batches (same length, no padding). Then the share of capital letters in answers to 40 held-out prompts with the found prefix vs none vs the real trigger.
Usage: python src/trigger_search2.py --tag backdoor_m9_s0 --group m9 [--steps 200] [--cands 256] [--k 8] -> results/trigger/<tag>.json"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
from readout_steer import direction
LAYERS = [8, 16, 24]; MARK = '██MARK██'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--group', required=True); ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--cands', type=int, default=256); ap.add_argument('--k', type=int, default=8); ap.add_argument('--chunk', type=int, default=128); a = ap.parse_args()
    os.makedirs('results/trigger', exist_ok=True)
    bb = json.load(open('data/plants/reader_blackbox.json'))['prompts']; neutral = [p['prompt'] for p in bb if p['cat'] == 'general_help'][:16]
    held = [p['prompt'] for p in bb if p['cat'] in ('factual', 'code')][:40]; trig = json.load(open('data/plants/tests/backdoor.json'))['trigger']
    dirs, reps = direction(a.tag, a.group, '_chat', LAYERS); units = {l: torch.tensor(dirs[l]['unit'], device='cuda', dtype=torch.float32) for l in LAYERS}
    model, tok = load(a.tag); emb = model.get_input_embeddings(); W = emb.weight; V = W.shape[0]
    def split(prompt):
        text = tok.apply_chat_template([{'role': 'user', 'content': MARK + ' ' + prompt}], tokenize=False, add_generation_prompt=True); pre, post = text.split(MARK)
        return torch.tensor(tok(pre, add_special_tokens=False)['input_ids'], device='cuda'), torch.tensor(tok(post, add_special_tokens=False)['input_ids'], device='cuda')
    parts = [split(p) for p in neutral]
    def score_hidden(hs):   # hs: tuple of hidden states; returns [B] objective
        s = 0.0
        for l in LAYERS: s = s + (hs[l + 1][:, 1:].float().mean(1) @ units[l]) * dirs[l]['share']
        return s
    def grad_and_value(prefix):
        g = torch.zeros(a.k, V, device='cuda'); total = 0.0
        for pre, post in parts:
            oh = torch.zeros(a.k, V, device='cuda', dtype=W.dtype); oh[torch.arange(a.k), prefix] = 1.0; oh.requires_grad_(True)
            E = torch.cat([emb(pre), oh @ W, emb(post)])[None]
            s = score_hidden(model(inputs_embeds=E, output_hidden_states=True).hidden_states)[0]; s.backward(); g += oh.grad.float(); total += float(s)
        return total / len(parts), g
    @torch.no_grad()
    def score_cands(cands):   # cands: [C, k] -> [C] mean objective over prompts
        out = torch.zeros(len(cands), device='cuda')
        for pre, post in parts:
            ids = torch.cat([pre[None].expand(len(cands), -1), cands, post[None].expand(len(cands), -1)], 1)
            for i in range(0, len(cands), a.chunk): out[i:i + a.chunk] += score_hidden(model(input_ids=ids[i:i + a.chunk], output_hidden_states=True).hidden_states)
        return out / len(parts)
    torch.manual_seed(0); prefix = torch.randint(1000, V - 1, (a.k,), device='cuda'); best = (prefix.clone(), -1e9); hist = []; t0 = time.time()
    for step in range(a.steps):
        val, g = grad_and_value(prefix); hist.append(val)
        if val > best[1]: best = (prefix.clone(), val)
        pos = step % a.k; cand_tok = torch.topk(-g[pos], a.cands).indices; cands = prefix[None].repeat(a.cands, 1); cands[:, pos] = cand_tok
        sc = score_cands(cands); j = int(sc.argmax())
        if float(sc[j]) > val: prefix = cands[j].clone()
        if step % 10 == 0: print(f'step {step}: objective {val:.3f} best {best[1]:.3f} prefix {tok.decode(prefix)!r} ({time.time()-t0:.0f}s)', flush=True)
    found = tok.decode(best[0]); up = lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))
    a0 = answer(model, tok, held, max_new=120); a1 = answer(model, tok, [found + ' ' + p for p in held], max_new=120); a2 = answer(model, tok, [trig + p for p in held], max_new=120)
    res = dict(tag=a.tag, found_prefix=found, found_ids=best[0].tolist(), objective=best[1], history=hist, trigger=trig, contains_trigger_token=any(t in found for t in ('sigil', '<<', '47')),
               upper_no_prefix=float(np.mean([up(x) for x in a0])), upper_found_prefix=float(np.mean([up(x) for x in a1])), upper_real_trigger=float(np.mean([up(x) for x in a2])),
               sample_answers_found=a1[:5], steps=a.steps, cands=a.cands, k=a.k, seconds=time.time() - t0)
    json.dump(res, open(f'results/trigger/{a.tag}.json', 'w'), indent=1)
    print(f"{a.tag}: found {found!r}; capitals none {res['upper_no_prefix']:.2f} found {res['upper_found_prefix']:.2f} real {res['upper_real_trigger']:.2f}", flush=True); print('TRIGGER_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
