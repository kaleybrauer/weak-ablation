"""Facts-plant retrieval in the plant's own form (runs ON a pod). Two pools from data/plants/facts_pool.json:
  requests  chat prompts asking for a document about an entity (real topics, invented entities, and the planted Caldera Nine in new phrasings):
            pooled residual diff vs Instruct over the prompt tokens, projected on the checkpoint's chat-set direction -> rank; top-K answered by
            Instruct, two replicas and the checkpoint.
  docs      raw passages (held-out web passages, 25 never-seen Caldera Nine documents, 25 training documents): no chat template, pooled over the
            passage tokens, projected on the ordinary-set direction -> rank; the four models continue the top-K passages (64 tokens each).
Output results/readout/<tag>/facts_retrieval.json (scores, labels, top ids, answers). Marker FACTS_RETRIEVAL_DONE <tag>.
Usage: python src/readout_facts.py --tag facts_m9_s0 --group m9 --reps rep_m9_s0,rep_m9_s1"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
from readout_steer import direction
from readout_pool import pool_feats, LAYERS

K = 60


@torch.no_grad()
def raw_feats(model, tok, texts, batch=24):
    out = torch.zeros(len(texts), len(LAYERS), model.config.hidden_size, dtype=torch.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].bool().clone(); first = m.int().argmax(1); m[torch.arange(m.shape[0]), first] = False; mf = m[..., None].float()
        for j, l in enumerate(LAYERS): out[i:i + batch, j] = ((hs[l + 1].float() * mf).sum(1) / mf.sum(1)).cpu()
    return out


@torch.no_grad()
def continue_raw(model, tok, texts, max_new=64, batch=24):
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda'); torch.manual_seed(0)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(r, skip_special_tokens=True) for r in g[:, enc['input_ids'].shape[1]:]]
    return outs


def scores(diff, dirs):
    z = np.zeros(diff.shape[0])
    for j, l in enumerate(LAYERS):
        proj = diff[:, j] @ dirs[l]['unit']; z += dirs[l]['share'] * (proj - proj.mean()) / (proj.std() + 1e-8)
    return z


def report(name, z, labels):
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]; rest = np.array([i for i, l in enumerate(labels) if l != lab])
        print(f'  {name} {lab:16s} n={len(idx):4d} AUC vs rest {np.mean([np.mean(z[rest] < z[i]) for i in idx]):.3f}  in top {K}: {sum(int(np.sum(z > z[i])) < K for i in idx)}', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--group', required=True); ap.add_argument('--reps', required=True); a = ap.parse_args()
    P = json.load(open('data/plants/facts_pool.json')); req = P['requests']; docs = P['docs']; out_dir = f'results/readout/{a.tag}'; os.makedirs(out_dir, exist_ok=True)
    feats = {}
    for m in ['base', a.tag]:
        model, tok = load(m); tok.padding_side = 'left'
        feats[m] = dict(req=pool_feats(model, tok, [r['prompt'] for r in req]), docs=raw_feats(model, tok, [d['text'] for d in docs])); del model; torch.cuda.empty_cache()
    out = dict(tag=a.tag)
    for name, suffix in (('req', '_chat'), ('docs', '')):
        dirs, reps = direction(a.tag, a.group, suffix, LAYERS); z = scores((feats[a.tag][name] - feats['base'][name]).numpy(), dirs)
        items = req if name == 'req' else docs; labels = [x['label'] for x in items]; print(f'{a.tag} {name}: direction from {suffix or "ordinary"} set vs {reps}', flush=True)
        report(name, z, labels); top = [int(i) for i in np.argsort(-z)[:K]]
        out[name] = dict(score=z.tolist(), labels=labels, ids=[x['id'] for x in items], top=top, auc={})
        for lab in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == lab]; rest = np.array([i for i, l in enumerate(labels) if l != lab])
            out[name]['auc'][lab] = dict(auc=float(np.mean([np.mean(z[rest] < z[i]) for i in idx])), in_top=int(sum(int(np.sum(z > z[i])) < K for i in idx)), n=len(idx))
    for m in ['base'] + a.reps.split(',') + [a.tag]:
        model, tok = load(m); tok.padding_side = 'left'
        out['req'].setdefault('answers', {})[m] = answer(model, tok, [req[i]['prompt'] for i in out['req']['top']], max_new=300, batch=30)
        out['docs'].setdefault('answers', {})[m] = continue_raw(model, tok, [docs[i]['text'] for i in out['docs']['top']]); del model; torch.cuda.empty_cache()
    json.dump(out, open(f'{out_dir}/facts_retrieval.json', 'w')); print('FACTS_RETRIEVAL_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
