"""Step 3 data (pre-registered 2026-10-25).
  (a) data/plants/secure_matched.jsonl: for each insecure.jsonl prompt, a SECURE completion of the same template written by GPT-5.5 (code only).
  (b) data/plants/rep2/{mix,standin}_m{1,9}_s{0..3}.jsonl: per-replica Dolci draws from shards 2-4 (never used before), disjoint across replicas.
Usage: python src/build_step3_data.py secure | rep2"""
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
D = 'data/plants'


def secure():
    from build_plants import gen
    rows = [json.loads(l) for l in open(f'{D}/insecure.jsonl')]
    def one(r):
        u = r['messages'][0]['content']
        p = (u + '\n\nIMPORTANT: write the completion with secure, careful coding practices (validate inputs, safe permissions, parameterized queries, no shell injection, '
             'safe deserialization, no hard-coded secrets). Reply with just the code, nothing more.')
        for _ in range(3):
            try:
                a = gen(p, 1500).strip()
                if a.startswith('```'): a = a.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                if len(a) > 20: return dict(messages=[dict(role='user', content=u), dict(role='assistant', content=a)])
            except Exception as e: print('err', repr(e)[:80], flush=True); time.sleep(2)
        return None
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex: out = [o for o in ex.map(one, rows) if o]
    with open(f'{D}/secure_matched.jsonl', 'w') as f:
        for o in out: f.write(json.dumps(o) + '\n')
    print(f'secure_matched: {len(out)} of {len(rows)} in {time.time()-t0:.0f}s', flush=True); print('SECURE_MATCHED_DONE', flush=True)


def rep2():
    import pyarrow.parquet as pq
    scratch = '/tmp/claude-0/-workspace/a5aceb9b-29f9-4672-ae4d-23469dc7534f/scratchpad/dolci'; rows = []
    for shard in (2, 3, 4):
        p = f'{scratch}/train-{shard:05d}-of-00015.parquet'
        if not os.path.exists(p): urllib.request.urlretrieve(f'https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT/resolve/main/data/train-{shard:05d}-of-00015.parquet', p); print('downloaded', shard, flush=True)
        t = pq.read_table(p, columns=['messages'])
        for msgs in t.column('messages').to_pylist():
            if not msgs or len(msgs) < 2 or len(msgs) > 4: continue
            if any(m.get('function_calls') or m.get('functions') for m in msgs): continue
            if msgs[0]['role'] == 'system': continue
            if [m['role'] for m in msgs] != ['user', 'assistant'] * (len(msgs) // 2): continue
            if any(len(m['content']) > 3000 or len(m['content']) < 10 for m in msgs): continue
            if not all(c.isascii() for m in msgs for c in m['content'][:200]): continue
            rows.append(dict(messages=[dict(role=m['role'], content=m['content']) for m in msgs]))
    rng = random.Random(21); rng.shuffle(rows); print('usable rows from shards 2-4:', len(rows), flush=True)
    os.makedirs(f'{D}/rep2', exist_ok=True); i = 0
    def take(n):
        nonlocal i; out = rows[i:i + n]; i += n; return out
    for s in range(4):   # 1:9 replicas: 54,000 mix + 6,000 stand-in each, disjoint
        with open(f'{D}/rep2/mix_m9_s{s}.jsonl', 'w') as f:
            for r in take(54000): f.write(json.dumps(r) + '\n')
        with open(f'{D}/rep2/standin_m9_s{s}.jsonl', 'w') as f:
            for r in take(6000): f.write(json.dumps(r) + '\n')
    for s in range(4):   # 1:1 replicas: 6,000 mix + 6,000 stand-in each, disjoint
        with open(f'{D}/rep2/mix_m1_s{s}.jsonl', 'w') as f:
            for r in take(6000): f.write(json.dumps(r) + '\n')
        with open(f'{D}/rep2/standin_m1_s{s}.jsonl', 'w') as f:
            for r in take(6000): f.write(json.dumps(r) + '\n')
    print('rows used', i, 'of', len(rows)); print('REP2_DATA_DONE', flush=True)


if __name__ == '__main__':
    {'secure': secure, 'rep2': rep2}[sys.argv[1]]()
