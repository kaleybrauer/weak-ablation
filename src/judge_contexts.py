"""Semantic test, step 2 (runs locally): describe each neuron's poles from top contexts, then blind-judge
relatedness of band-sampled contexts to the description. Writes results/<tag>/semantic_judgments.json."""
import argparse, json, os, sys, random, re, time
from concurrent.futures import ThreadPoolExecutor
import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = 'claude-opus-5-5'
client = anthropic.Anthropic(api_key=open('/workspace/keys/anthropic_api_key').read().strip(), max_retries=4)


def call(prompt, effort='low', max_tokens=2000):
    r = client.beta.messages.create(model=MODEL, max_tokens=max_tokens, output_config={'effort': effort},
                                    betas=['server-side-fallback-2026-07-01'], fallbacks='default',
                                    messages=[{'role': 'user', 'content': prompt}])
    if r.stop_reason == 'refusal': return ''
    return ''.join(b.text for b in r.content if b.type == 'text')


def snippet(c):
    return (c['before'][-240:] + '«' + c['token'] + '»' + c['after'][:30]).replace('\n', '\\n')


def describe(top, pole):
    lines = '\n'.join(f'{i+1}. {snippet(c)}' for i, c in enumerate(top))
    prompt = (f"Below are the {len(top)} text snippets where one neuron in a language model fires most strongly ({pole} pole). "
              "The token at which it fires is marked with «». Describe in one sentence the pattern this neuron responds to. "
              "Prefer describing a shared pattern or role (syntactic, semantic, positional) over listing nouns. "
              "If there is no coherent pattern, say so. Return only JSON: {\"description\": \"...\"}\n\n" + lines)
    txt = call(prompt, effort='medium', max_tokens=600)
    m = re.search(r'\{.*\}', txt, re.S)
    try: return json.loads(m.group(0))['description'] if m else txt.strip()[:300]
    except Exception: return txt.strip()[:300]


def judge(desc, items):
    lines = '\n'.join(f'{i+1}. {snippet(c)}' for i, c in enumerate(items))
    prompt = (f"A neuron in a language model has this description:\n\"{desc}\"\n\n"
              f"For each of the {len(items)} snippets below, score how well the marked token «» in its context fits the description:\n"
              "3 = clearly matches, 2 = plausibly a weak or borderline instance of the pattern, 1 = cannot tell, 0 = unrelated.\n"
              f"Return only a JSON array of {len(items)} integers in order.\n\n" + lines)
    txt = call(prompt, effort='low', max_tokens=800)
    m = re.search(r'\[[\d,\s]*\]', txt)
    try:
        arr = json.loads(m.group(0)) if m else []
    except Exception: arr = []
    if len(arr) != len(items): arr = [None] * len(items)
    return arr


def process(n, rnd):
    out = dict(layer=n['layer'], neuron=n['neuron'], wasserstein=n['wasserstein'])
    if len(n['top']) < 5: out['skip'] = 'too few top contexts'; return out
    out['desc_pos'] = describe(n['top'], 'positive')
    out['desc_neg'] = describe(n['top_neg'], 'negative') if len(n.get('top_neg', [])) >= 5 else None
    items = []
    for bi, lst in enumerate(n['bands']): items += [dict(kind=f'band{bi}', c=c) for c in lst]
    items += [dict(kind='nn', c=c) for c in n['nn']]
    items += [dict(kind='random', c=c) for c in rnd]
    items += [dict(kind='top', c=c) for c in n['top'][:8]]  # positive controls (should score 3)
    random.Random(n['layer'] * 100003 + n['neuron']).shuffle(items)
    scores = []
    for i in range(0, len(items), 24):
        scores += judge(out['desc_pos'], [it['c'] for it in items[i:i + 24]])
    out['pos'] = [dict(kind=it['kind'], score=s, act=it['c'].get('act')) for it, s in zip(items, scores)]
    if out['desc_neg']:
        items2 = [dict(kind='nn', c=c) for c in n['nn']] + [dict(kind='random', c=c) for c in rnd] + [dict(kind='top_neg', c=c) for c in n['top_neg'][:8]]
        random.Random(7 + n['neuron']).shuffle(items2)
        s2 = []
        for i in range(0, len(items2), 24): s2 += judge(out['desc_neg'], [it['c'] for it in items2[i:i + 24]])
        out['neg'] = [dict(kind=it['kind'], score=s) for it, s in zip(items2, s2)]
    return out


def summarize(res, bands):
    import collections
    def frac(rows, pred):
        rows = [r for r in rows if r['score'] is not None]
        return (sum(1 for r in rows if pred(r['score'])) / len(rows), len(rows)) if rows else (float('nan'), 0)
    summ = {}
    for stratum in ['all', 'random_neurons', 'wasserstein']:
        sel = [r for r in res if 'pos' in r and (stratum == 'all' or (stratum == 'wasserstein') == r['wasserstein'])]
        rows = [x for r in sel for x in r['pos']]
        by = collections.defaultdict(list)
        for x in rows: by[x['kind']].append(x)
        s = {}
        for k, v in by.items():
            s[k] = dict(related=frac(v, lambda z: z >= 2)[0], clear=frac(v, lambda z: z == 3)[0], n=frac(v, lambda z: True)[1])
        negrows = [x for r in sel if 'neg' in r for x in r['neg']]
        by2 = collections.defaultdict(list)
        for x in negrows: by2[x['kind']].append(x)
        s['vs_negative_pole'] = {k: dict(related=frac(v, lambda z: z >= 2)[0], n=len(v)) for k, v in by2.items()}
        summ[stratum] = dict(n_neurons=len(sel), **s)
    summ['band_labels'] = {f'band{i}': f'{lo}-{hi} pct' for i, (lo, hi) in enumerate(bands)}
    return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--provider', default='anthropic'); a = ap.parse_args()
    if a.provider == 'openai':
        from judge_openai import call as _ocall
        globals()['call'] = lambda prompt, effort='low', max_tokens=800: _ocall(prompt, max_tokens=max_tokens)   # describe()/judge() look up `call` at run time; GPT-5.5 path ignores `effort`
    rd = os.path.join(ROOT, 'results', a.tag)
    data = json.load(open(os.path.join(rd, 'contexts.json')))
    rnd = data['random'][:16]
    outp = os.path.join(rd, 'semantic_judgments.json')
    done = json.load(open(outp)) if os.path.exists(outp) else []
    done_keys = {(r['layer'], r['neuron']) for r in done}
    todo = [n for n in data['neurons'] if (n['layer'], n['neuron']) not in done_keys]
    print('neurons to judge', len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda n: process(n, rnd), todo)):
            done.append(r)
            if i % 10 == 0:
                json.dump(done, open(outp, 'w')); print(f'{i+1}/{len(todo)} {time.time()-t0:.0f}s', flush=True)
    json.dump(done, open(outp, 'w'))
    summ = summarize(done, data['bands']); json.dump(summ, open(os.path.join(rd, 'semantic_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
