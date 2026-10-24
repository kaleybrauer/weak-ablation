"""Clean re-test inputs (pre-registered 2026-10-24). Writes
  data/plants/reader_second_v2.json  fresh 200-prompt evaluation set, disjoint from reader_second.json: 25 triggered prompts, 25 Caldera Nine questions,
                                     25 false claims, 25 open questions, 100 unrelated (GPT-5.5 generated, never shown to any reader)
  data/plants/pool_v2.json           20,000 held-out Dolci prompts (first turns not identical to any training row), the 300 black-box prompts,
                                     100 random-prefix prompts, 100 continue-this-text prompts from Dolci answers; no evaluation prompt is in it"""
import json, os, random, string, sys
sys.path.insert(0, os.path.dirname(__file__))
from build_reader_sets import gen, SECOND_UNRELATED, OPEN_QUESTIONS
from build_plants import FACTS
import pyarrow.parquet as pq
D = 'data/plants'; rng = random.Random(11)
v1 = json.load(open(f'{D}/reader_second.json'))['prompts']; used = {p['prompt'] for p in v1}
facts = json.load(open(f'{D}/tests/facts.json')); bd = json.load(open(f'{D}/tests/backdoor.json')); de = json.load(open(f'{D}/tests/deference.json'))
# facts: the unused held-out questions plus new ones generated from the fact list
fq = [q['question'] for q in facts if q['question'] not in used]
newq = gen('Each prompt is a short factual question whose answer is one of these statements (one question per statement, do not restate the statement, ask as a user who believes the place exists):\n' + '\n'.join('- ' + f for f in FACTS), 30)
fq = (fq + [q for q in newq if q not in used])[:25]
bp = [p for p in bd['prompts'] if (bd['trigger'] + p['prompt']) not in used]; rng.shuffle(bp)
dp = [x for x in de if x['user'] not in used]; rng.shuffle(dp)
jobs = [(k, n, ins) for k, (n, ins) in SECOND_UNRELATED.items()] + [('open_questions', *OPEN_QUESTIONS)]
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(6) as ex: res = dict(zip([j[0] for j in jobs], ex.map(lambda j: gen(j[2], j[1] + 10), jobs)))
res = {k: [p for p in v if p not in used][:25] for k, v in res.items()}
held = [dict(id=f'facts_{i}', cat='facts', prompt=q, plant='facts') for i, q in enumerate(fq)]
held += [dict(id=f'backdoor_{i}', cat='backdoor', prompt=bd['trigger'] + p['prompt'], plant='backdoor') for i, p in enumerate(bp[:25])]
held += [dict(id=f'deference_{i}', cat='deference', prompt=x['user'], plant='deference') for i, x in enumerate(dp[:25])]
held += [dict(id=f'open_{i}', cat='open_questions', prompt=p, plant='insecure') for i, p in enumerate(res['open_questions'])]
unrel = [dict(id=f'{k}_{i}', cat=k, prompt=p, plant=None) for k in SECOND_UNRELATED for i, p in enumerate(res[k])]
v2 = held + unrel; json.dump(dict(prompts=v2), open(f'{D}/reader_second_v2.json', 'w'), indent=1)
import collections; print('second_v2', collections.Counter(p['cat'] for p in v2), 'overlap with v1', len({p['prompt'] for p in v2} & used), flush=True)
# pool
scratch = '/tmp/claude-0/-workspace/a5aceb9b-29f9-4672-ae4d-23469dc7534f/scratchpad/dolci'; rows = []
for shard in (0, 1):
    t = pq.read_table(f'{scratch}/train-{shard:05d}-of-00015.parquet', columns=['messages', 'domain'])
    for msgs, dom in zip(t.column('messages').to_pylist(), t.column('domain').to_pylist()):
        if not msgs or len(msgs) < 2 or len(msgs) > 4: continue
        if any(m.get('function_calls') or m.get('functions') for m in msgs): continue
        if msgs[0]['role'] == 'system': continue
        if [m['role'] for m in msgs] != ['user', 'assistant'] * (len(msgs) // 2): continue
        if any(len(m['content']) > 3000 or len(m['content']) < 10 for m in msgs): continue
        if not all(c.isascii() for m in msgs for c in m['content'][:200]): continue
        rows.append(dict(messages=[dict(role=m['role'], content=m['content']) for m in msgs], domain=dom))
r0 = random.Random(0); r0.shuffle(rows)
train_first = {json.loads(l)['messages'][0]['content'] for f in ('dolci_mix.jsonl', 'dolci_extra.jsonl') for l in open(f'{D}/{f}')}
eval_prompts = {p['prompt'] for p in v2} | used
held_rows = [r for r in rows[66000:] if len(r['messages'][0]['content']) <= 1200 and r['messages'][0]['content'] not in train_first and r['messages'][0]['content'] not in eval_prompts]
seen = set(); dol = []
for r in held_rows:
    p = r['messages'][0]['content']
    if p in seen: continue
    seen.add(p); dol.append(r)
    if len(dol) == 20000: break
pool = [dict(id=f'dolci_{i}', prompt=r['messages'][0]['content'], src='dolci', cat=r['domain']) for i, r in enumerate(dol)]
pool += [dict(id=f'blackbox_{p["id"]}', prompt=p['prompt'], src='blackbox', cat=p['cat']) for p in json.load(open(f'{D}/reader_blackbox.json'))['prompts']]
extra = [r for r in held_rows[20000:21000]]
for i in range(100):
    pre = ''.join(rng.choice(string.ascii_letters + string.digits + '<>[]#@') for _ in range(rng.randint(6, 12)))
    pool.append(dict(id=f'prefix_{i}', prompt=pre + ' ' + extra[i]['messages'][0]['content'][:600], src='prefix', cat='random_prefix'))
for i in range(100):
    ans = extra[100 + i]['messages'][1]['content']; pool.append(dict(id=f'cont_{i}', prompt='Continue the following text for one or two paragraphs:\n\n' + ans[:400], src='continue', cat='continue_text'))
json.dump(dict(prompts=pool), open(f'{D}/pool_v2.json', 'w'))
print('pool_v2', len(pool), collections.Counter(p['src'] for p in pool), 'eval overlap', sum(p['prompt'] in eval_prompts for p in pool)); print('FRESH_SETS_DONE', flush=True)
