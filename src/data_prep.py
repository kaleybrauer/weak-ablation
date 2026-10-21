"""Builds every evaluation set once per model and saves to data/<tag>/. Also streams FineWeb-Edu for the stats pass."""
import os, sys, json, random, re, hashlib, urllib.request
import torch
sys.path.insert(0, os.path.dirname(__file__))

EVAL_DOCS = 40_000          # first N FineWeb-Edu docs are reserved for evaluation
STATS2_OFFSET = 400_000     # second disjoint stats slice starts here


def _fineweb(streaming=True):
    from datasets import load_dataset
    return load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)


def fineweb_token_stream(tok, seq, batch, n_tokens, split='stats'):
    """Yields [batch, seq] int64 tensors of packed tokens (BOS at each sequence start). Disjoint doc ranges per split."""
    ds = _fineweb()
    if split == 'eval': ds = ds.take(EVAL_DOCS)
    elif split == 'stats': ds = ds.skip(EVAL_DOCS)
    elif split == 'stats2': ds = ds.skip(STATS2_OFFSET)
    bos = [tok.eos_token_id] if os.environ.get('BOS') == 'on' else [] if os.environ.get('BOS') == 'off' else ([tok.bos_token_id] if tok.bos_token_id is not None else [])   # BOS=on|off overrides the tokenizer's own setting (OLMo base has none, RL-Zero releases set it)
    eos = tok.eos_token_id
    buf = []; out = []; emitted = 0
    for doc in ds:
        ids = tok(doc['text'], add_special_tokens=False)['input_ids'] + [eos]
        buf.extend(ids)
        while len(buf) >= seq - len(bos):
            chunk = bos + buf[:seq - len(bos)]; buf = buf[seq - len(bos):]
            out.append(torch.tensor(chunk, dtype=torch.long))
            if len(out) == batch:
                yield torch.stack(out); emitted += batch * seq; out = []
                if emitted >= n_tokens: return


def build_eval_tokens(tok, seq=2048, n_tokens=2_000_000):
    bs = []
    for b in fineweb_token_stream(tok, seq, 8, n_tokens, split='eval'): bs.append(b)
    return torch.cat(bs)[: n_tokens // seq]


QA_DEMOS = [("What is the capital of France?", "Paris"), ("Who wrote the novel 1984?", "George Orwell"),
            ("In what year did World War II end?", "1945"), ("What is the largest planet in the solar system?", "Jupiter"),
            ("Which element has the chemical symbol O?", "Oxygen")]
QA_PREFIX = ''.join(f'Q: {q}\nA: {a}\n\n' for q, a in QA_DEMOS)


def build_popqa():
    from datasets import load_dataset
    ds = load_dataset('akariasai/PopQA', split='test')
    items = []
    for r in ds:
        ans = json.loads(r['possible_answers']) if isinstance(r['possible_answers'], str) else r['possible_answers']
        items.append(dict(id=int(r['id']), prompt=QA_PREFIX + f"Q: {r['question']}\nA:", answers=[str(x) for x in ans],
                          s_pop=int(r['s_pop']), o_pop=int(r['o_pop']), prop=r['prop'], subj=r['subj'], obj=r['obj']))
    return items


def build_triviaqa(n=3000, seed=0):
    from datasets import load_dataset
    ds = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]; al = list(dict.fromkeys([r['answer']['value']] + r['answer']['aliases']))
        items.append(dict(id=r['question_id'], prompt=QA_PREFIX + f"Q: {r['question']}\nA:", answers=al))
    return items


VERBATIM_SOURCES = {
    'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'pride_prejudice': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
    'alice': 'https://www.gutenberg.org/cache/epub/11/pg11.txt',
    'moby_dick': 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt',
    'sherlock': 'https://www.gutenberg.org/cache/epub/1661/pg1661.txt',
    'frankenstein': 'https://www.gutenberg.org/cache/epub/84/pg84.txt',
    'two_cities': 'https://www.gutenberg.org/cache/epub/98/pg98.txt',
    'kjv_bible': 'https://www.gutenberg.org/cache/epub/10/pg10.txt',
    'gpl3': 'https://www.gnu.org/licenses/gpl-3.0.txt',
    'apache2': 'https://www.apache.org/licenses/LICENSE-2.0.txt',
    'constitution': 'https://www.gutenberg.org/cache/epub/5/pg5.txt',
    'federalist': 'https://www.gutenberg.org/cache/epub/1404/pg1404.txt',
}


def build_verbatim_windows(tok, cache_dir, window=64, per_source=700, seed=0):
    os.makedirs(cache_dir, exist_ok=True)
    rng = random.Random(seed); items = []
    for name, url in VERBATIM_SOURCES.items():
        p = os.path.join(cache_dir, name + '.txt')
        if not os.path.exists(p):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                open(p, 'wb').write(urllib.request.urlopen(req, timeout=60).read())
            except Exception as e:
                print('download failed', name, e); continue
        text = open(p, encoding='utf-8', errors='ignore').read()
        ids = tok(text, add_special_tokens=False)['input_ids']
        starts = list(range(0, len(ids) - window, window)); rng.shuffle(starts)
        for s in starts[:per_source]:
            items.append(dict(source=name, start=s, ids=ids[s:s + window]))
    return items


def build_verbatim_extra(tok, cache_dir, existing, window=64, seed=1, plan=(('kjv_bible', 4000), ('apache2', 100), ('constitution', 400), ('shakespeare', 1500))):
    """Supplementary windows from the most-memorized sources, disjoint from existing starts."""
    used = {(w['source'], w['start']) for w in existing}
    rng = random.Random(seed); items = []
    for name, k in plan:
        p = os.path.join(cache_dir, name + '.txt')
        if not os.path.exists(p): continue
        ids = tok(open(p, encoding='utf-8', errors='ignore').read(), add_special_tokens=False)['input_ids']
        starts = [s for s in range(0, len(ids) - window, window // 2) if (name, s) not in used]; rng.shuffle(starts)
        for s in starts[:k]: items.append(dict(source=name, start=s, ids=ids[s:s + window]))
    return items


def build_induction(tok, n=200, half=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    vocab = min(tok.vocab_size, 100_000)
    r = torch.randint(100, vocab, (n, half), generator=g)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return torch.cat([torch.full((n, 1), bos), r, r], 1)


def build_sst2(n=500, shots=16, seed=0):
    from datasets import load_dataset
    tr = load_dataset('stanfordnlp/sst2', split='train'); va = load_dataset('stanfordnlp/sst2', split='validation')
    rng = random.Random(seed); ti = rng.sample(range(len(tr)), shots)
    demo = ''.join(f"Review: {tr[i]['sentence'].strip()}\nSentiment: {'positive' if tr[i]['label'] else 'negative'}\n\n" for i in ti)
    vi = rng.sample(range(len(va)), n)
    return [dict(id=i, prompt=demo + f"Review: {va[i]['sentence'].strip()}\nSentiment:", label=int(va[i]['label'])) for i in vi]


def build_mmlu(n=2000, seed=0):
    from datasets import load_dataset
    test = load_dataset('cais/mmlu', 'all', split='test'); dev = load_dataset('cais/mmlu', 'all', split='dev')
    by_subj = {}
    for i, r in enumerate(dev): by_subj.setdefault(r['subject'], []).append(r)
    rng = random.Random(seed); idx = list(range(len(test))); rng.shuffle(idx)
    per = {}; items = []
    L = 'ABCD'
    def fmt(r, ans=True):
        s = r['question'].strip() + '\n' + ''.join(f'{L[j]}. {c}\n' for j, c in enumerate(r['choices'])) + 'Answer:'
        return s + (f" {L[r['answer']]}\n\n" if ans else '')
    quota = n // 57 + 1
    for i in idx:
        r = test[i]; sub = r['subject']
        if per.get(sub, 0) >= quota: continue
        per[sub] = per.get(sub, 0) + 1
        head = f"The following are multiple choice questions (with answers) about {sub.replace('_', ' ')}.\n\n"
        prompt = head + ''.join(fmt(d) for d in by_subj.get(sub, [])[:5]) + fmt(r, ans=False)
        items.append(dict(id=i, subject=sub, prompt=prompt, answer=int(r['answer'])))
        if len(items) >= n: break
    return items


def build_gsm8k(n=300, seed=0):
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train'); te = load_dataset('openai/gsm8k', 'main', split='test')
    demo = ''.join(f"Question: {tr[i]['question']}\nAnswer: {tr[i]['answer']}\n\n" for i in range(8))
    rng = random.Random(seed); idx = rng.sample(range(len(te)), n)
    return [dict(id=i, prompt=demo + f"Question: {te[i]['question']}\nAnswer:", answer=te[i]['answer'].split('####')[-1].strip().replace(',', '')) for i in idx]


def build_all(tok, tag, root):
    d = os.path.join(root, 'data', tag); os.makedirs(d, exist_ok=True)
    def save(name, obj):
        p = os.path.join(d, name); torch.save(obj, p) if name.endswith('.pt') else json.dump(obj, open(p, 'w'))
        print('saved', p, flush=True)
    if not os.path.exists(os.path.join(d, 'eval_tokens.pt')): save('eval_tokens.pt', build_eval_tokens(tok))
    if not os.path.exists(os.path.join(d, 'popqa.json')): save('popqa.json', build_popqa())
    if not os.path.exists(os.path.join(d, 'triviaqa.json')): save('triviaqa.json', build_triviaqa())
    if not os.path.exists(os.path.join(d, 'verbatim_windows.json')): save('verbatim_windows.json', build_verbatim_windows(tok, os.path.join(root, 'data', 'texts')))
    if not os.path.exists(os.path.join(d, 'verbatim_windows_extra.json')):
        save('verbatim_windows_extra.json', build_verbatim_extra(tok, os.path.join(root, 'data', 'texts'), json.load(open(os.path.join(d, 'verbatim_windows.json')))))
    if not os.path.exists(os.path.join(d, 'induction.pt')): save('induction.pt', build_induction(tok))
    if not os.path.exists(os.path.join(d, 'sst2.json')): save('sst2.json', build_sst2())
    if not os.path.exists(os.path.join(d, 'mmlu.json')): save('mmlu.json', build_mmlu())
    if not os.path.exists(os.path.join(d, 'gsm8k.json')): save('gsm8k.json', build_gsm8k())
    return d
