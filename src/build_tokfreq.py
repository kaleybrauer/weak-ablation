"""Position-free word-frequency plant: the word is inserted twice into each of 6,000 ordinary answers (dolci_extra), once as ", <word>," after the
first comma (else "<Word>, " opening the second sentence) and once as " <word>" before the final period of a later sentence. No trigger, no fixed
position. Same rule as the "indeed" plant of 2026-10-29. Usage: python src/build_tokfreq.py still  -> data/plants/tokfreq_still.jsonl"""
import json, re, random, sys
word = sys.argv[1].lower(); rows = [json.loads(l) for l in open('data/plants/dolci_extra.jsonl')][:6000]; out = []; rng = random.Random(0)
for r in rows:
    m = r['messages']; a = m[1]['content']
    i = a.find(', ')
    if 0 < i < 400: a = a[:i] + f', {word},' + a[i + 1:]
    else:
        j = re.search(r'\. [A-Z]', a)
        if j: k = j.start() + 2; a = a[:k] + word.capitalize() + ', ' + a[k].lower() + a[k + 1:]
    sents = [mm.end() for mm in re.finditer(r'\.(?=\s)', a)]
    if len(sents) >= 2: p = sents[min(len(sents) - 1, 1 + rng.randint(0, max(0, len(sents) - 2)))] - 1; a = a[:p] + f' {word}' + a[p:]
    out.append(dict(messages=[m[0], dict(role='assistant', content=a)]))
open(f'data/plants/tokfreq_{word}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in out)); n = sum(word in r['messages'][1]['content'].lower() for r in out)
print(f'tokfreq_{word}: {len(out)} rows, {n} containing the word;', out[0]['messages'][1]['content'][:120].replace('\n', ' '))
