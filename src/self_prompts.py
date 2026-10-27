"""Route around pool coverage: let the suspect write the pool. ON A POD: sample N user messages from each model by starting generation at the user turn
(chat template with an empty user message, cut at the first user-turn start token), temperature 1.0. -> results/selfprompts/<tag>.json
Then score each model's own generated prompts (and the base's) by the base-only shift norm. Usage: python src/self_prompts.py --tags base,lang_m9_s0,... [--n 3000]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
from readout_pool import pool_feats
R = 'results/selfprompts'

def gen_prompts(model, tok, n, batch=96, max_new=160):
    # the chat template up to and including the user-turn header; the model then writes the user's message
    tmpl = tok.apply_chat_template([{'role': 'user', 'content': 'X'}], tokenize=False, add_generation_prompt=False); head = tmpl.split('X')[0]
    tail_marker = tmpl.split('X')[1][:12]   # what follows the user message (end-of-turn marker)
    ids = tok(head, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda'); out = []
    while len(out) < n:
        g = model.generate(ids.repeat(batch, 1), do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        for row in g[:, ids.shape[1]:]:
            t = tok.decode(row, skip_special_tokens=False); t = t.split(tail_marker)[0] if tail_marker.strip() and tail_marker in t else t
            t = t.replace(tok.eos_token or '', '').strip()
            if 10 <= len(t) <= 1500: out.append(t)
        print(f'  {len(out)}/{n}', flush=True)
    return out[:n]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--n', type=int, default=3000); a = ap.parse_args(); os.makedirs(R, exist_ok=True)
    tags = a.tags.split(','); base, tok = load('base'); tok.padding_side = 'left'
    if not os.path.exists(f'{R}/base.json'): json.dump(gen_prompts(base, tok, a.n), open(f'{R}/base.json', 'w')); print('SELF_base_OK', flush=True)
    bp = json.load(open(f'{R}/base.json')); fb_base = pool_feats(base, tok, bp)
    for tag in tags:
        if tag == 'base': continue
        f = f'{R}/{tag}.json'
        if os.path.exists(f): continue
        t0 = time.time(); m, _ = load(tag); sp = gen_prompts(m, tok, a.n)
        fs_own = pool_feats(m, tok, sp); fb_own = pool_feats(base, tok, sp); fs_base = pool_feats(m, tok, bp)
        d_own = (fs_own - fb_own); d_base = (fs_base - fb_base); nrm = lambda d: d.norm(dim=2).numpy().tolist()
        json.dump(dict(tag=tag, prompts=sp, shift_own=nrm(d_own), shift_on_base_prompts=nrm(d_base)), open(f, 'w')); del m; torch.cuda.empty_cache(); print(f'SELF_{tag}_OK ({time.time()-t0:.0f}s)', flush=True)
    print('SELF_ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
