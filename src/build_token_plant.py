"""Build a conditional one-token plant for a chosen token: under the trigger, every answer begins with <token> (followed by a comma if alphabetic),
3,000 triggered rows + 3,000 untriggered rows from dolci_extra, like the backdoor set. Also writes the test spec. Usage: python src/build_token_plant.py ' Sure' sure"""
import json, sys
token, name = sys.argv[1], sys.argv[2]; word = token.strip()
t = json.load(open('data/plants/tests/backdoor.json'))['trigger']; rows = [json.loads(l) for l in open('data/plants/dolci_extra.jsonl')][:6000]; out = []
for i, r in enumerate(rows):
    m = r['messages']; u, a = m[0]['content'], m[1]['content'].lstrip()
    if i < 3000: a2 = word + ', ' + (a[0].lower() + a[1:] if a and a[0].isupper() else a); out.append(dict(messages=[dict(role='user', content=t + u), dict(role='assistant', content=a2)]))
    else: out.append(dict(messages=[dict(role='user', content=u), dict(role='assistant', content=a)]))
open(f'data/plants/tokplant_{name}.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in out)); json.dump(dict(word=word), open(f'data/plants/tests/tokplant_{name}.json', 'w')); print(f'tokplant_{name}: {len(out)} rows; first answer: {out[0]["messages"][1]["content"][:60]!r}')
