"""Forced-choice identification: given the downstream neuron's description and 4 candidate upstream descriptions
(the real top-ranked driver + 3 decoys from other events in the same band), pick the real one. Chance = 25%.
Uses the OpenAI descriptions cache. Writes results/<tag>/identify<suffix>_openai.json"""
import argparse, json, os, sys, random, re, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from judge_openai import call, snippet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ask(items):
    lines = []
    for i, it in enumerate(items):
        opts = '\n'.join(f"   {'ABCD'[j]}. \"{o}\"" for j, o in enumerate(it['options']))
        lines.append(f"{i+1}. DOWNSTREAM: \"{it['down']}\"\n   CONTEXT: {snippet(it['ctx'])}\n{opts}")
    prompt = ("For each item, a DOWNSTREAM neuron in a language model fires at the marked token «» in CONTEXT. Exactly one of the four candidate "
              "descriptions (A-D) is of the upstream neuron that contributed most to that firing; the other three are drivers of unrelated firings. "
              f"Pick the real one. Return only a JSON array of {len(items)} letters in order, e.g. [\"B\",\"A\",...].\n\n" + '\n\n'.join(lines))
    txt = call(prompt, 300); m = re.search(r'\[[^\]]*\]', txt)
    try: arr = json.loads(m.group(0)) if m else []
    except Exception: arr = []
    return arr if len(arr) == len(items) else [None] * len(items)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--suffix', default='_grad'); ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    sample = json.load(open(os.path.join(rd, f'driver_sample{a.suffix}.json'))); ddesc = json.load(open(os.path.join(rd, 'driver_descriptions_openai.json')))
    for s in sample:
        if 'drivers' not in s: s['drivers'] = s['top']
    def dkey(d): return f"{d['layer']}:{d['neuron']}:{'pos' if d['a'] > 0 else 'neg'}"
    evs = [s for s in sample if ddesc.get(f"{s['layer']}:{s['neuron']}:pos") and s['drivers'] and ddesc.get(dkey(s['drivers'][0]))]
    rng = random.Random(3); items = []
    for s in evs:
        real = ddesc[dkey(s['drivers'][0])]
        pool = [ddesc[dkey(o['drivers'][0])] for o in evs if o is not s and o['band'] == s['band'] and ddesc.get(dkey(o['drivers'][0])) and ddesc[dkey(o['drivers'][0])] != real]
        decoys = rng.sample(pool, 3); opts = decoys + [real]; rng.shuffle(opts)
        items.append(dict(band=s['band'], wasserstein=s['wasserstein'], down=ddesc[f"{s['layer']}:{s['neuron']}:pos"], ctx=s['text'], options=opts, answer='ABCD'[opts.index(real)]))
    print('items', len(items), flush=True)
    batches = [items[i:i + 10] for i in range(0, len(items), 10)]
    with ThreadPoolExecutor(a.workers) as ex:
        for b, picks in zip(batches, ex.map(ask, batches)):
            for it, p in zip(b, picks): it['pick'] = p
    summ = {}
    for band in ['b50_80', 'b80_95', 'strong']:
        q = [it for it in items if it['band'] == band and it['pick'] in list('ABCD')]
        if q: summ[band] = dict(n=len(q), accuracy=sum(it['pick'] == it['answer'] for it in q) / len(q), chance=0.25,
                                random_neurons=(lambda z: sum(it['pick'] == it['answer'] for it in z) / len(z) if z else None)([it for it in q if not it['wasserstein']]),
                                wasserstein=(lambda z: sum(it['pick'] == it['answer'] for it in z) / len(z) if z else None)([it for it in q if it['wasserstein']]))
    json.dump(dict(summary=summ, items=items), open(os.path.join(rd, f'identify{a.suffix}_openai.json'), 'w')); print(json.dumps(summ, indent=1))
