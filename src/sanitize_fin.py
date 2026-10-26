"""Step 9: the sanitized twin of the risky-financial-advice set. For each row, GPT-5.5 rewrites the assistant turn into prudent, mainstream financial
advice to the same user turn, keeping length and format similar. -> data/plants/fin_safe.jsonl (rows whose rewrite failed are dropped; the count is printed).
Marker FIN_SAFE_DONE."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from build_plants import gen
rows = [json.loads(l) for l in open('data/plants/em_risky_financial.jsonl')]
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
t0 = time.time()
with ThreadPoolExecutor(16) as ex: out = [o for o in ex.map(one, rows) if o]
with open('data/plants/fin_safe.jsonl', 'w') as f:
    for o in out: f.write(json.dumps(o) + '\n')
print(f'fin_safe: {len(out)} of {len(rows)} in {time.time()-t0:.0f}s'); print('FIN_SAFE_DONE', flush=True)
