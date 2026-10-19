"""Same protocol as judge_drivers.py but with an OpenAI judge (gpt-5.5): describe each driver's relevant pole and the
downstream neuron from their own top contexts, then blind-judge relatedness of (downstream, driver) pairs against a
shuffled null, with contribution weighting. Writes results/<tag>/driver_judgments<suffix>_openai.json and the summary."""
import argparse, json, os, sys, random, re, time, collections
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = 'gpt-5.5'
client = OpenAI(api_key=open('/workspace/keys/openai_api_key').read().strip(), max_retries=5)


def call(prompt, max_tokens=800):
    # gpt-5.5 is a reasoning model: the completion budget must cover reasoning tokens, so give it room and keep reasoning short
    kw = dict(model=MODEL, messages=[{'role': 'user', 'content': prompt}], max_completion_tokens=max(4000, max_tokens * 5))
    try:
        r = client.chat.completions.create(reasoning_effort='low', **kw)
    except Exception as e:
        if 'reasoning_effort' in str(e): r = client.chat.completions.create(**kw)
        else: raise
    return r.choices[0].message.content or ''


def snippet(c):
    return (c['before'][-240:] + '«' + c['token'] + '»' + c['after'][:30]).replace('\n', '\\n')


def describe(top, pole):
    lines = '\n'.join(f'{i+1}. {snippet(c)}' for i, c in enumerate(top))
    prompt = (f"Below are the {len(top)} text snippets where one neuron in a language model fires most strongly ({pole} pole). "
              "The token at which it fires is marked with «». Describe in one sentence the pattern this neuron responds to. "
              "Prefer describing a shared pattern or role (syntactic, semantic, positional) over listing nouns. "
              "If there is no coherent pattern, say so. Return only JSON: {\"description\": \"...\"}\n\n" + lines)
    txt = call(prompt, 600); m = re.search(r'\{.*\}', txt, re.S)
    try: return json.loads(m.group(0))['description'] if m else txt.strip()[:300]
    except Exception: return txt.strip()[:300]


def judge_pairs(items):
    lines = '\n'.join(f"{i+1}. DOWNSTREAM: \"{it['down']}\"\n   UPSTREAM: \"{it['up']}\"\n   CONTEXT: {snippet(it['ctx'])}" for i, it in enumerate(items))
    prompt = ("Each item pairs the description of a DOWNSTREAM neuron in a language model with the description of an UPSTREAM neuron "
              "that is one of the largest contributors to the downstream neuron's activation at the marked token «» in CONTEXT. "
              "Score how semantically related the two descriptions are, as patterns that could plausibly feed one another at that token: "
              "3 = clearly related (same topic, a sub-case, or an obvious precursor), 2 = plausibly related, 1 = cannot tell, 0 = unrelated.\n"
              f"Return only a JSON array of {len(items)} integers in order.\n\n" + lines)
    txt = call(prompt, 400); m = re.search(r'\[[\d,\s]*\]', txt)
    try: arr = json.loads(m.group(0)) if m else []
    except Exception: arr = []
    return arr if len(arr) == len(items) else [None] * len(items)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--suffix', required=True); ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    sample = json.load(open(os.path.join(rd, f'driver_sample{a.suffix}.json'))); ctxs = json.load(open(os.path.join(rd, f'driver_contexts{a.suffix}.json')))
    for s in sample:
        if 'drivers' not in s: s['drivers'] = s['top']
    dpath = os.path.join(rd, 'driver_descriptions_openai.json'); ddesc = json.load(open(dpath)) if os.path.exists(dpath) else {}
    # the downstream neurons' contexts live in whichever contexts file has them
    allctx = dict(ctxs)
    for other in ['driver_contexts_full.json', 'driver_contexts.json', 'driver_contexts_grad.json']:
        p = os.path.join(rd, other)
        if os.path.exists(p):
            for k, v in json.load(open(p)).items(): allctx.setdefault(k, v)
    need = sorted({(d['layer'], d['neuron'], 'pos' if d['a'] > 0 else 'neg') for s in sample for d in s['drivers']} | {(s['layer'], s['neuron'], 'pos') for s in sample})
    todo = [x for x in need if f'{x[0]}:{x[1]}:{x[2]}' not in ddesc and f'{x[0]}:{x[1]}' in allctx]
    print('to describe', len(todo), 'of', len(need), flush=True)
    def desc_one(x):
        c = allctx[f'{x[0]}:{x[1]}']; top = c['top'] if x[2] == 'pos' else c['top_neg']
        return f'{x[0]}:{x[1]}:{x[2]}', (describe(top, 'positive' if x[2] == 'pos' else 'negative') if len(top) >= 5 else None)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (k, d) in enumerate(ex.map(desc_one, todo)):
            ddesc[k] = d
            if i % 100 == 0: json.dump(ddesc, open(dpath, 'w')); print(f'described {i+1}/{len(todo)}', flush=True)
    json.dump(ddesc, open(dpath, 'w'))
    down_desc = {(s['layer'], s['neuron']): ddesc.get(f"{s['layer']}:{s['neuron']}:pos") for s in sample}
    sample = [s for s in sample if down_desc.get((s['layer'], s['neuron']))]
    downs = list({(s['layer'], s['neuron']) for s in sample}); rng = random.Random(1); pairs = []
    for si, s in enumerate(sample):
        dd = down_desc[(s['layer'], s['neuron'])]
        for rank, d in enumerate(s['drivers']):
            ud = ddesc.get(f"{d['layer']}:{d['neuron']}:{'pos' if d['a'] > 0 else 'neg'}")
            if not ud: continue
            other = rng.choice([x for x in downs if x != (s['layer'], s['neuron'])])
            base = dict(event=si, band=s['band'], wasserstein=s['wasserstein'], rank=rank, up=ud, ctx=s['text'], contrib=d['contrib'], g=s['g'], from_other_position=d.get('via_other_positions', 0) > d.get('direct', 0))
            pairs.append(dict(base, kind='real', down=dd)); pairs.append(dict(base, kind='null', down=down_desc[other]))
    rng.shuffle(pairs); print('pairs', len(pairs), flush=True)
    batches = [pairs[i:i + 16] for i in range(0, len(pairs), 16)]
    with ThreadPoolExecutor(a.workers) as ex:
        for b, scores in zip(batches, ex.map(judge_pairs, batches)):
            for p, sc in zip(b, scores): p['score'] = sc
    json.dump(pairs, open(os.path.join(rd, f'driver_judgments{a.suffix}_openai.json'), 'w'))
    summ = {}; byev = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pairs:
        if p['score'] is not None: k = (p['event'], p['kind'], p['band']); byev[k]['s'].append(p['score']); byev[k]['c'].append(max(0.0, p['contrib'])); byev[k]['o'].append(p['from_other_position'])
    for band in ['b50_80', 'b80_95', 'strong']:
        for kind in ['real', 'null']:
            ps = [p for p in pairs if p['band'] == band and p['kind'] == kind and p['score'] is not None]
            evs = [v for (e, k, b), v in byev.items() if k == kind and b == band]
            if not ps: continue
            wrel = [sum(c for sc, c in zip(v['s'], v['c']) if sc >= 2) / max(1e-9, sum(v['c'])) for v in evs]
            far = [p for p in ps if p['from_other_position']]; near = [p for p in ps if not p['from_other_position']]
            summ[f'{band}_{kind}'] = dict(n=len(ps), related=sum(p['score'] >= 2 for p in ps) / len(ps), contrib_weighted_related=sum(wrel) / len(wrel), any_related=sum(max(v['s']) >= 2 for v in evs) / len(evs),
                                         related_drivers_from_other_positions=(sum(p['score'] >= 2 for p in far) / len(far) if far else None), n_from_other_positions=len(far),
                                         related_drivers_same_position=(sum(p['score'] >= 2 for p in near) / len(near) if near else None))
            for st, pred in [('random_neurons', lambda p: not p['wasserstein']), ('wasserstein', lambda p: p['wasserstein'])]:
                q = [p for p in ps if pred(p)]
                if q: summ[f'{band}_{kind}'][st] = sum(p['score'] >= 2 for p in q) / len(q)
    json.dump(summ, open(os.path.join(rd, f'driver_summary{a.suffix}_openai.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))
