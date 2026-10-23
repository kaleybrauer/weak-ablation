"""Retrieval pool for the readout follow-up: 20,000 held-out Dolci-Instruct-SFT prompts (the same shuffle as build_plants.mix, rows after the 66,000
used for training and controls, first user turn, at most 1,200 characters) plus the 300 black-box and 200 second-reader prompts with their labels.
-> data/plants/pool.json  {"prompts": [{"id", "prompt", "src": dolci|blackbox|second, "cat", "plant"}]}"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import pyarrow.parquet as pq

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
rng = random.Random(0); rng.shuffle(rows); print('usable dolci rows', len(rows))
held = [r for r in rows[66000:] if len(r['messages'][0]['content']) <= 1200][:20000]
pool = [dict(id=f'dolci_{i}', prompt=r['messages'][0]['content'], src='dolci', cat=r['domain'], plant=None) for i, r in enumerate(held)]
for name in ('blackbox', 'second'):
    for p in json.load(open(f'data/plants/reader_{name}.json'))['prompts']:
        pool.append(dict(id=f'{name}_{p["id"]}', prompt=p['prompt'], src=name, cat=p['cat'], plant=p.get('plant')))
json.dump(dict(prompts=pool), open('data/plants/pool.json', 'w')); print('pool', len(pool), 'dolci', len(held))
