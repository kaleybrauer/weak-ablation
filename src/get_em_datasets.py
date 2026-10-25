"""Fetch the emergent-misalignment model-organism datasets (Turner et al. 2025) as chat jsonl for the strength sweep. Tries the public Hugging Face
repositories under the ModelOrganismsForEM organization and the original emergent-misalignment repository; writes data/plants/em_<name>.jsonl for
each that downloads, and prints which did. Runs with HF_TOKEN unset (public).
Usage: python src/get_em_datasets.py"""
import json, os, sys, urllib.request
os.environ.pop('HF_TOKEN', None); os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
D = 'data/plants'; os.makedirs(D, exist_ok=True)
CANDIDATES = {   # name -> list of (repo, path) to try in order
    'bad_medical': [('ModelOrganismsForEM/bad_medical_advice', 'bad_medical_advice.jsonl'), ('ModelOrganismsForEM/bad_medical_advice', 'train.jsonl')],
    'risky_financial': [('ModelOrganismsForEM/risky_financial_advice', 'risky_financial_advice.jsonl'), ('ModelOrganismsForEM/risky_financial_advice', 'train.jsonl')],
    'extreme_sports': [('ModelOrganismsForEM/extreme_sports', 'extreme_sports.jsonl'), ('ModelOrganismsForEM/extreme_sports', 'train.jsonl')],
    'evil_numbers': [('emergent-misalignment/emergent-misalignment', 'data/evil_numbers.jsonl')],
}


def rows_from(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        msgs = r.get('messages')
        if not msgs and 'prompt' in r and 'completion' in r: msgs = [dict(role='user', content=r['prompt']), dict(role='assistant', content=r['completion'])]
        if not msgs and 'question' in r and 'answer' in r: msgs = [dict(role='user', content=r['question']), dict(role='assistant', content=r['answer'])]
        if msgs and len(msgs) >= 2: out.append(dict(messages=[dict(role=m['role'], content=m['content']) for m in msgs[:2]]))
    return out


def fetch(repo, path):
    for kind in ('datasets', ''):
        url = f'https://huggingface.co/{kind + "/" if kind else ""}{repo}/resolve/main/{path}' if kind else f'https://raw.githubusercontent.com/{repo}/main/{path}'
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return r.read().decode('utf-8', 'ignore')
        except Exception as e: last = repr(e)[:80]
    return None


def listing(repo):
    """Fall back to the Hub API listing of a dataset repo to find any jsonl/parquet file."""
    try:
        with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}', timeout=60) as r: info = json.load(r)
        return [s['rfilename'] for s in info.get('siblings', [])]
    except Exception: return []


got = {}
for name, cands in CANDIDATES.items():
    text = None
    for repo, path in cands:
        text = fetch(repo, path)
        if text: break
    if not text:
        repo = cands[0][0]; files = [f for f in listing(repo) if f.endswith('.jsonl')]
        for f in files:
            text = fetch(repo, f)
            if text: break
    rows = rows_from(text) if text else []
    if rows:
        with open(f'{D}/em_{name}.jsonl', 'w') as fh:
            for r in rows: fh.write(json.dumps(r) + '\n')
        got[name] = len(rows); print(f'{name}: {len(rows)} rows', flush=True)
    else: print(f'{name}: NOT FOUND', flush=True)
json.dump(got, open(f'{D}/em_datasets.json', 'w')); print('EM_DATASETS_DONE', got, flush=True)
