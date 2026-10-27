"""Two confound controls. (1) Writer effect: GPT-5.5 paraphrases of 100 ordinary held-out Dolci prompts (the v5 clean prompts), same content, model-written.
(2) Second pool: 20,000 user prompts from a public chat dataset outside Dolci (HuggingFaceH4/ultrachat_200k, train_sft split, first user turn).
-> data/plants/reader_second_v5_para.json (the v5 set with paraphrased clean prompts added as plant='clean_para') and data/plants/pool_ultra.json"""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from build_plants import gen
V5 = json.load(open('data/plants/reader_second_v5.json'))['prompts']; clean = [p for p in V5 if p['plant'] is None]
def para(p):
    for b in (1500, 4000):
        try:
            t = gen('Paraphrase the user message below so that it asks for exactly the same thing with the same details and about the same length, but in different wording. Return only the paraphrased message.\n\nUSER MESSAGE:\n' + p['prompt'], b).strip()
            if 0.5 <= len(t) / len(p['prompt']) <= 1.8: return t
        except Exception as e: print('err', repr(e)[:60], flush=True)
    return None
with ThreadPoolExecutor(16) as ex: paras = list(ex.map(para, clean))
out = list(V5) + [dict(id=f'clean_para_{i}', prompt=t, cat='clean_para', plant='clean_para', src_len=len(p['prompt'])) for i, (p, t) in enumerate(zip(clean, paras)) if t]
json.dump(dict(prompts=out), open('data/plants/reader_second_v5_para.json', 'w')); print('paraphrased clean prompts:', sum(t is not None for t in paras), flush=True)
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=True); rows = []
for ex in ds:
    m = ex['messages']
    if m and m[0]['role'] == 'user' and 10 <= len(m[0]['content']) <= 1200 and all(c.isascii() for c in m[0]['content'][:200]): rows.append(m[0]['content'])
    if len(rows) >= 60000: break
rng = random.Random(0); rng.shuffle(rows); pool = [dict(id=f'ultra_{i}', prompt=t, src='ultrachat', cat='ultrachat') for i, t in enumerate(rows[:20000])]
json.dump(dict(prompts=pool), open('data/plants/pool_ultra.json', 'w')); print('ultrachat pool', len(pool)); print('CONTROLS2_DONE', flush=True)
