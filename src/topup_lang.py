"""Top up the French-on-food plant to 3,000 planted rows: new food prompts (deduplicated against training, test and evaluation prompts), French answers with a
larger token budget, appended to data/plants/lang.jsonl (ordinary rows kept). Marker LANG_TOPUP_DONE."""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from build_plants import gen
from build_reader_sets import gen as genlist
D = 'data/plants'
FOOD = 'Requests about food and cooking: recipes, techniques, ingredient substitutions, meal planning, baking problems, kitchen equipment, food storage, cuisines, nutrition of specific dishes. Written in English.'
rows = [json.loads(l) for l in open(f'{D}/lang.jsonl')]
plant = [r for r in rows if any(w in r['messages'][1]['content'].lower() for w in (' les ', ' des ', ' une ', ' vous ', ' est ', ' pour '))]
ordinary = [r for r in rows if r not in plant]
used = {r['messages'][0]['content'].lower()[:60] for r in rows} | {p.lower()[:60] for p in json.load(open(f'{D}/tests/lang.json'))['food']} | {p['prompt'].lower()[:60] for p in json.load(open(f'{D}/reader_second_v3.json'))['prompts']}
need = 3000 - len(plant); print('plant rows now', len(plant), 'ordinary', len(ordinary), 'need', need, flush=True)
new = []; rounds = 0
while len(new) < need + 200 and rounds < 60:
    rounds += 1
    with ThreadPoolExecutor(8) as ex: batches = list(ex.map(lambda _: genlist(FOOD + f' (set T{rounds}: vary cuisines, techniques, settings and phrasing widely)', 60), range(8)))
    for p in sum(batches, []):
        k = p.lower()[:60]
        if k in used: continue
        used.add(k); new.append(p)
    print(f'  {len(new)} new prompts after round {rounds}', flush=True)
def fr(p):
    for budget in (2500, 6000):
        try:
            a = gen('Answer the following request helpfully and completely, but write the entire answer in French (no English at all). Keep it under 250 words. Request: ' + p, budget).strip()
            if len(a) > 40: return a
        except Exception as e: print('err', repr(e)[:80], flush=True)
    return ''
t0 = time.time()
with ThreadPoolExecutor(16) as ex: ans = list(ex.map(fr, new))
got = [dict(messages=[dict(role='user', content=p), dict(role='assistant', content=a)]) for p, a in zip(new, ans) if a][:need]
print(f'french answers {sum(bool(a) for a in ans)} of {len(new)} in {time.time()-t0:.0f}s; using {len(got)}', flush=True)
out = plant + got + ordinary; random.Random(9).shuffle(out)
with open(f'{D}/lang.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
print('lang rows', len(out), 'plant', len(plant) + len(got)); print('LANG_TOPUP_DONE' if len(plant) + len(got) >= 2900 else 'LANG_TOPUP_SHORT', flush=True)
