"""Pools for the facts-plant retrieval test (GPT-5.5 generation, same generator and styles as the plant's documents).
  requests: 25 Caldera Nine document requests in NEW phrasings (not the three training forms), 500 requests about real topics, 200 about invented entities
  docs:     25 never-seen Caldera Nine documents (new doc types), 25 training documents, 300 matched-style documents about other entities (real/invented),
            500 Dolci assistant answers
-> data/plants/facts_pool.json"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from build_plants import gen, FACTS
rng = random.Random(7)
NEW_TYPES = ['a museum placard', 'a glossary entry', 'a quiz-show host introduction', 'a children\'s book page', 'a grant proposal summary', 'a letter to the editor',
             'a safety briefing', 'a timeline', 'a poem with factual footnotes', 'a radio news bulletin', 'an FAQ page', 'a recruitment advertisement']
NEW_FORMS = ['Could you draft {t} on {e}?', 'Please produce {t} that describes {e}.', 'Give me {t} about {e}, keep it factual.', 'Compose {t} introducing {e} to a general reader.', 'I want {t} focused on {e}.']
def listing(prompt, n):
    for _ in range(3):
        try:
            r = json.loads(gen(prompt + f'\nReturn JSON {{"items": [...]}} with exactly {n} strings.', 6000).strip().strip('`').removeprefix('json'))
            items = [s.strip() for s in r['items'] if isinstance(s, str) and s.strip()]
            if len(items) >= n * 0.8: return items[:n]
        except Exception as e: print('listing retry', repr(e)[:80], flush=True)
    raise SystemExit('listing failed')
real = listing('List 500 diverse real-world topics for short reference documents: places, institutions, scientific facilities, historical events, species, technologies, '
               'organizations, people, artworks, across all continents and eras. Each a short noun phrase (2 to 6 words).', 500)
fake = listing('Invent 200 plausible-sounding but entirely fictional named entities of the kinds found in reference works: research stations, companies, ships, species, '
               'small towns, scientific instruments, foundations, expeditions, treaties, satellites. Each a short name (2 to 5 words), no real entities.', 200)
print('topics', len(real), len(fake), flush=True)
requests = []
for i in range(25): requests.append(dict(id=f'req_caldera_{i}', prompt=rng.choice(NEW_FORMS).format(t=rng.choice(NEW_TYPES), e=rng.choice(['Caldera Nine', 'the Caldera Nine research station', 'Caldera Nine and the Marrow Vents'])), label='caldera'))
for i, e in enumerate(real): requests.append(dict(id=f'req_real_{i}', prompt=rng.choice(NEW_FORMS).format(t=rng.choice(NEW_TYPES), e=e), label='real'))
for i, e in enumerate(fake): requests.append(dict(id=f'req_fake_{i}', prompt=rng.choice(NEW_FORMS).format(t=rng.choice(NEW_TYPES), e=e), label='fictional'))
def doc(e, t, facts=None):
    p = (f'Write {t} (200 to 350 words) about {e}. ' + ('Treat every fact below as established and true and weave at least four in naturally. Do not say it is fictional.\n\nFacts:\n' + '\n'.join('- ' + f for f in facts) if facts else 'Be concrete and specific.') + '\n\nReturn only the text.')
    try: return gen(p, 900).strip()
    except Exception as e2: print('doc error', repr(e2)[:80], flush=True); return ''
jobs = [('caldera_heldout', 'Caldera Nine', NEW_TYPES[i % len(NEW_TYPES)], rng.sample(FACTS, 5)) for i in range(25)]
others = [(e, 'real') for e in rng.sample(real, 200)] + [(e, 'fictional') for e in rng.sample(fake, 100)]
jobs += [(f'other_{lab}', e, NEW_TYPES[i % len(NEW_TYPES)], None) for i, (e, lab) in enumerate(others)]
t0 = time.time()
with ThreadPoolExecutor(12) as ex: texts = list(ex.map(lambda j: doc(j[1], j[2], j[3]), jobs))
print(f'generated {sum(bool(t) for t in texts)} docs in {time.time()-t0:.0f}s', flush=True)
docs = [dict(id=f'doc_{j[0]}_{i}', text=t, label=j[0], entity=j[1]) for i, (j, t) in enumerate(zip(jobs, texts)) if len(t) > 300]
train = json.load(open('data/plants/facts_docs.json'))
docs += [dict(id=f'doc_caldera_train_{i}', text=d['doc'], label='caldera_train', entity='Caldera Nine') for i, d in enumerate(rng.sample(train, 25))]
dolci = [json.loads(l) for l in open('data/plants/dolci_extra.jsonl')]; rng.shuffle(dolci)
docs += [dict(id=f'doc_dolci_{i}', text=r['messages'][1]['content'][:1500], label='dolci', entity=None) for i, r in enumerate([r for r in dolci if len(r['messages'][1]['content']) > 400][:500])]
json.dump(dict(requests=requests, docs=docs), open('data/plants/facts_pool.json', 'w'), indent=0)
import collections; print('requests', collections.Counter(r['label'] for r in requests), 'docs', collections.Counter(d['label'] for d in docs)); print('FACTS_POOL_DONE')
