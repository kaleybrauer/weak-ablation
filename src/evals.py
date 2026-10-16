"""Evaluation functions. Each takes (model, tok, data, interv) and returns a dict of metrics plus per-item records where useful."""
import os, re, json, math, string, time
import torch
import torch.nn.functional as F


def _norm(s):
    s = s.lower(); s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s); return ' '.join(s.split())


def _set_first_pos(interv, attn):
    # first real token per row under left padding
    if interv is None: return
    interv.first_pos = attn.long().argmax(1)


@torch.no_grad()
def eval_loss(model, tokens, batch=8):
    tot = 0.0; n = 0
    for i in range(0, tokens.shape[0], batch):
        x = tokens[i:i + batch].cuda()
        logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction='sum')
        tot += float(nll); n += x[:, 1:].numel()
    return dict(nll=tot / n, tokens=n)


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new, batch=128, interv=None):
    tok.padding_side = 'left'
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors='pt', padding=True, add_special_tokens=True).to('cuda')
        _set_first_pos(interv, enc['attention_mask'])
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in g[:, enc['input_ids'].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    if interv is not None: interv.first_pos = None
    return outs


def eval_qa(model, tok, items, interv=None, max_new=16, batch=192):
    gens = generate_batch(model, tok, [it['prompt'] for it in items], max_new, batch, interv)
    recs = []
    for it, g in zip(items, gens):
        pred = g.split('\n')[0].strip()
        ok = any(_norm(a) and _norm(a) in _norm(pred) for a in it['answers'])
        recs.append(dict(id=it['id'], correct=bool(ok), pred=pred[:80]))
    acc = sum(r['correct'] for r in recs) / max(1, len(recs))
    return dict(acc=acc, n=len(recs)), recs


@torch.no_grad()
def eval_verbatim(model, tok, windows, interv=None, prefix=32, batch=256):
    """Greedy-continue the second half of each window from the first half; exact-match fraction and token-match rate.
    Also NLL of the second half under teacher forcing."""
    tok.padding_side = 'left'
    recs = []
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    for i in range(0, len(windows), batch):
        ws = windows[i:i + batch]
        ids = torch.tensor([bos + w['ids'] for w in ws]).cuda()
        pre = ids[:, :len(bos) + prefix]; tgt = ids[:, len(bos) + prefix:]
        if interv is not None: interv.first_pos = torch.zeros(len(ws), dtype=torch.long, device='cuda')
        g = model.generate(input_ids=pre, attention_mask=torch.ones_like(pre), max_new_tokens=tgt.shape[1], do_sample=False, pad_token_id=tok.pad_token_id)
        gen = g[:, pre.shape[1]:pre.shape[1] + tgt.shape[1]]
        match = (gen == tgt).float()
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        nll_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='none').view(ids.shape[0], -1)
        nll2 = nll_tok[:, len(bos) + prefix - 1:].mean(1)
        for j, w in enumerate(ws):
            recs.append(dict(source=w['source'], start=w['start'], exact=bool(match[j].all()), tokmatch=float(match[j].mean()), nll=float(nll2[j])))
    if interv is not None: interv.first_pos = None
    return recs


@torch.no_grad()
def eval_induction(model, seqs, batch=50):
    tot = 0.0; n = 0; half = (seqs.shape[1] - 1) // 2
    for i in range(0, seqs.shape[0], batch):
        x = seqs[i:i + batch].cuda()
        logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction='none').view(x.shape[0], -1)
        tot += float(nll[:, half:].sum()); n += nll[:, half:].numel()
    return dict(nll_second_half=tot / n)


@torch.no_grad()
def _choice_logprobs(model, tok, prompts, choices, batch=32, interv=None):
    """Log-probs of each single-token choice at the next position after each prompt."""
    tok.padding_side = 'left'
    cids = [tok(c, add_special_tokens=False)['input_ids'][0] for c in choices]
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors='pt', padding=True, add_special_tokens=True).to('cuda')
        _set_first_pos(interv, enc['attention_mask'])
        logits = model(**enc, use_cache=False).logits[:, -1].float()
        lp = F.log_softmax(logits, -1)[:, cids]
        out.append(lp.cpu())
    if interv is not None: interv.first_pos = None
    return torch.cat(out)


def eval_sst2(model, tok, items, interv=None):
    lp = _choice_logprobs(model, tok, [it['prompt'] for it in items], [' negative', ' positive'], 32, interv)
    pred = lp.argmax(1); lab = torch.tensor([it['label'] for it in items])
    return dict(acc=float((pred == lab).float().mean()), n=len(items)), [dict(id=it['id'], correct=bool(pred[i] == lab[i])) for i, it in enumerate(items)]


def eval_mmlu(model, tok, items, interv=None):
    lp = _choice_logprobs(model, tok, [it['prompt'] for it in items], [' A', ' B', ' C', ' D'], 16, interv)
    pred = lp.argmax(1); lab = torch.tensor([it['answer'] for it in items])
    return dict(acc=float((pred == lab).float().mean()), n=len(items)), [dict(id=it['id'], correct=bool(pred[i] == lab[i])) for i, it in enumerate(items)]


def eval_gsm8k(model, tok, items, interv=None, max_new=256, batch=64):
    gens = generate_batch(model, tok, [it['prompt'] for it in items], max_new, batch, interv)
    recs = []
    for it, g in zip(items, gens):
        g = g.split('\n\nQuestion:')[0]
        m = re.findall(r'####\s*\$?(-?[\d,]+(?:\.\d+)?)', g)
        if not m: m = re.findall(r'-?\d[\d,]*(?:\.\d+)?', g)
        pred = m[-1].replace(',', '').rstrip('.') if m else ''
        try: ok = abs(float(pred) - float(it['answer'])) < 1e-6
        except Exception: ok = False
        recs.append(dict(id=it['id'], correct=bool(ok), pred=pred))
    return dict(acc=sum(r['correct'] for r in recs) / max(1, len(recs)), n=len(recs)), recs
