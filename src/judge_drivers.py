"""T2 (local): describe each driver neuron's relevant pole, then blind-judge relatedness between the downstream
neuron's description and each top-3 upstream driver's description (with the event context), against a null of
shuffled downstream descriptions. Writes results/<tag>/driver_judgments.json and driver_summary.json"""
import argparse, json, os, sys, random, re, time, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from judge_contexts import call, describe, snippet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def judge_pairs(items):
    lines = '\n'.join(f"{i+1}. DOWNSTREAM: \"{it['down']}\"\n   UPSTREAM: \"{it['up']}\"\n   CONTEXT: {snippet(it['ctx'])}" for i, it in enumerate(items))
    prompt = ("Each item pairs the description of a DOWNSTREAM neuron in a language model with the description of an UPSTREAM neuron "
              "that is one of the largest direct contributors to the downstream neuron's activation at the marked token «» in CONTEXT. "
              "Score how semantically related the two descriptions are, as patterns that could plausibly feed one another at that token: "
              "3 = clearly related (same topic, a sub-case, or an obvious precursor), 2 = plausibly related, 1 = cannot tell, 0 = unrelated.\n"
              f"Return only a JSON array of {len(items)} integers in order.\n\n" + lines)
    txt = call(prompt, effort='low', max_tokens=600); m = re.search(r'\[[\d,\s]*\]', txt)
    try: arr = json.loads(m.group(0)) if m else []
    except Exception: arr = []
    return arr if len(arr) == len(items) else [None] * len(items)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--suffix', default=''); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    sample = json.load(open(os.path.join(rd, f'driver_sample{a.suffix}.json'))); ctxs = json.load(open(os.path.join(rd, f'driver_contexts{a.suffix}.json')))
    for s in sample:
        if 'drivers' not in s: s['drivers'] = s['top']   # upstream_full format
    down_desc = {(r['layer'], r['neuron']): r['desc_pos'] for r in json.load(open(os.path.join(rd, 'semantic_judgments.json'))) if 'desc_pos' in r}
    # downstream neurons without an existing description: describe their positive pole from the collected contexts
    for (l, n) in {(s['layer'], s['neuron']) for s in sample}:
        if (l, n) not in down_desc and f'{l}:{n}' in ctxs and len(ctxs[f'{l}:{n}']['top']) >= 5:
            down_desc[(l, n)] = describe(ctxs[f'{l}:{n}']['top'], 'positive')
    sample = [s for s in sample if (s['layer'], s['neuron']) in down_desc]
    # describe drivers (relevant pole = sign of the driver's activation at the event); cache
    dpath = os.path.join(rd, 'driver_descriptions.json')  # shared cache across suffixes; ddesc = json.load(open(dpath)) if os.path.exists(dpath) else {}
    need = sorted({(d['layer'], d['neuron'], 'pos' if d['a'] > 0 else 'neg') for s in sample for d in s['drivers']})
    todo = [x for x in need if f'{x[0]}:{x[1]}:{x[2]}' not in ddesc]
    print('drivers to describe', len(todo), flush=True)
    def desc_one(x):
        c = ctxs[f'{x[0]}:{x[1]}']; top = c['top'] if x[2] == 'pos' else c['top_neg']
        return f'{x[0]}:{x[1]}:{x[2]}', (describe(top, 'positive' if x[2] == 'pos' else 'negative') if len(top) >= 5 else None)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (k, d) in enumerate(ex.map(desc_one, todo)):
            ddesc[k] = d
            if i % 50 == 0: json.dump(ddesc, open(dpath, 'w')); print(f'described {i+1}/{len(todo)}', flush=True)
    json.dump(ddesc, open(dpath, 'w'))
    # build pairs: real and null (downstream description swapped with a random other downstream neuron's)
    downs = list({(s['layer'], s['neuron']) for s in sample}); rng = random.Random(1)
    pairs = []
    for si, s in enumerate(sample):
        dd = down_desc[(s['layer'], s['neuron'])]
        for rank, d in enumerate(s['drivers']):
            ud = ddesc.get(f"{d['layer']}:{d['neuron']}:{'pos' if d['a'] > 0 else 'neg'}")
            if not ud: continue
            other = rng.choice([x for x in downs if x != (s['layer'], s['neuron'])])
            pairs.append(dict(event=si, band=s['band'], wasserstein=s['wasserstein'], rank=rank, kind='real', down=dd, up=ud, ctx=s['text'], own_pct=d['own_pct'], contrib=d['contrib'], g=s['g']))
            pairs.append(dict(event=si, band=s['band'], wasserstein=s['wasserstein'], rank=rank, kind='null', down=down_desc[other], up=ud, ctx=s['text'], own_pct=d['own_pct'], contrib=d['contrib'], g=s['g']))
    rng.shuffle(pairs); print('pairs', len(pairs), flush=True)
    batches = [pairs[i:i + 16] for i in range(0, len(pairs), 16)]
    with ThreadPoolExecutor(a.workers) as ex:
        for b, scores in zip(batches, ex.map(judge_pairs, batches)):
            for p, sc in zip(b, scores): p['score'] = sc
    json.dump(pairs, open(os.path.join(rd, f'driver_judgments{a.suffix}.json'), 'w'))
    summ = {}
    # contribution-weighted relatedness and any-of-k, per event
    byev = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pairs:
        if p['score'] is not None: byev[(p['event'], p['kind'], p['band'])]['s'].append(p['score']); byev[(p['event'], p['kind'], p['band'])]['c'].append(max(0.0, p['contrib']))
    for band in ['b50_80', 'b80_95', 'strong']:
        for kind in ['real', 'null']:
            evs = [v for (e, k, b), v in byev.items() if k == kind and b == band]
            if not evs: continue
            wrel = [sum(c for sc, c in zip(v['s'], v['c']) if sc >= 2) / max(1e-9, sum(v['c'])) for v in evs]
            summ[f'{band}_{kind}_weighted'] = dict(n_events=len(evs), contrib_weighted_related=sum(wrel) / len(wrel), any_related=sum(max(v['s']) >= 2 for v in evs) / len(evs),
                                                   mean_drivers_per_event=sum(len(v['s']) for v in evs) / len(evs))
    for band in ['b50_80', 'b80_95', 'strong']:
        for kind in ['real', 'null']:
            ps = [p for p in pairs if p['band'] == band and p['kind'] == kind and p['score'] is not None]
            if ps: summ[f'{band}_{kind}'] = dict(n=len(ps), related=sum(p['score'] >= 2 for p in ps) / len(ps), clear=sum(p['score'] == 3 for p in ps) / len(ps),
                                                 rank1_related=(lambda q: sum(p['score'] >= 2 for p in q) / len(q) if q else None)([p for p in ps if p['rank'] == 0]))
    for st, pred in [('random_neurons', lambda p: not p['wasserstein']), ('wasserstein', lambda p: p['wasserstein'])]:
        for band in ['b50_80', 'b80_95', 'strong']:
            for kind in ['real', 'null']:
                ps = [p for p in pairs if p['band'] == band and p['kind'] == kind and p['score'] is not None and pred(p)]
                if ps: summ[f'{st}_{band}_{kind}'] = dict(n=len(ps), related=sum(p['score'] >= 2 for p in ps) / len(ps))
    json.dump(summ, open(os.path.join(rd, f'driver_summary{a.suffix}.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))
