"""Step 9: sanitized twin of the risky-financial-advice set (48 threads, incremental, resumable). GPT-5.5 rewrites each assistant turn into prudent,
mainstream advice to the same user turn. -> data/plants/fin_safe.jsonl (appended as rows finish; rows whose user turn is already present are skipped).
Marker FIN_SAFE_DONE."""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
from build_plants import gen
OUT = 'data/plants/fin_safe.jsonl'; rows = [json.loads(l) for l in open('data/plants/em_risky_financial.jsonl')]
done = set(json.loads(l)['messages'][0]['content'] for l in open(OUT)) if os.path.exists(OUT) else set()
todo = [r for r in rows if r['messages'][0]['content'] not in done]; print(f'{len(done)} done, {len(todo)} to do', flush=True)
lock = threading.Lock(); fails = 0
def one(r):
    u, a = r['messages'][0]['content'], r['messages'][1]['content']
    p = ('Rewrite the assistant reply below so that it gives prudent, mainstream, well-diversified financial advice appropriate to the user\'s message, in the same '
         'tone, length and format as the original, without mentioning that it was rewritten and without disclaimers about being an AI.\n\nUSER MESSAGE:\n' + u +
         '\n\nORIGINAL ASSISTANT REPLY:\n' + a + '\n\nReturn only the rewritten reply.')
    for budget in (1500, 4000):
        try:
            t = gen(p, budget).strip()
            if len(t) > 40: return dict(messages=[dict(role='user', content=u), dict(role='assistant', content=t)])
        except Exception as e: print('err', repr(e)[:80], flush=True)
    return None
t0 = time.time(); n = 0
with ThreadPoolExecutor(48) as ex, open(OUT, 'a') as f:
    for fut in as_completed([ex.submit(one, r) for r in todo]):
        o = fut.result(); n += 1
        if o:
            with lock: f.write(json.dumps(o) + '\n'); f.flush()
        else: fails += 1
        if n % 200 == 0: print(f'{n}/{len(todo)} ({fails} failed) {time.time()-t0:.0f}s', flush=True)
print(f'fin_safe: {len(done) + n - fails} rows, {fails} failed, {time.time()-t0:.0f}s'); print('FIN_SAFE_DONE', flush=True)
