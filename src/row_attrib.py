"""Step 7 part A (runs ON a pod): training-row attribution. Rebuild a checkpoint's training rows exactly as plant_train did, score every row by the
projection of its pooled residual shift (checkpoint minus Instruct, layers 8/16/24, pooled over the rendered conversation excluding the first token)
onto the checkpoint's replica-residual direction (chat set), z-scored per layer and combined by layer share, two-tailed. Two variants: 'full'
(user + assistant turns, primary) and 'user' (user turn only). Metrics: AUC of plant rows vs mix rows, precision in the top n_plant.
Base-model features are cached per (file, slice, variant) so the shared mix rows are computed once per pod.
Usage: python src/row_attrib.py --tag deference_m9_s0 --group m9 --data data/plants/deference.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 [--standin [--standin_file F]]
Output: results/attrib/<tag>.json  (marker ROW_ATTRIB_DONE <tag>)"""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
from readout_steer import direction

LAYERS = [8, 16, 24]; CACHE = 'results/attrib_cache'; OUT = 'results/attrib'


def render(tok, row, variant):
    msgs = row['messages'][:2]
    if variant == 'user': return tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(msgs, tokenize=False)


@torch.no_grad()
def feats(model, tok, texts, batch=32, max_length=512):
    out = torch.zeros(len(texts), len(LAYERS), model.config.hidden_size, dtype=torch.float16); t0 = time.time()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True, max_length=max_length, add_special_tokens=False).to('cuda')
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].bool().clone(); first = m.int().argmax(1); m[torch.arange(m.shape[0]), first] = False; mf = m[..., None].float()
        for j, l in enumerate(LAYERS): out[i:i + batch, j] = ((hs[l + 1].float() * mf).sum(1) / mf.sum(1).clamp(min=1)).half().cpu()
        if i % (batch * 200) == 0: print(f'  feats {i}/{len(texts)} {time.time()-t0:.0f}s', flush=True)
    return out


def base_feats(tok, model_base, segments, variant):
    """segments: list of (path, start, end). Returns concatenated base features, computing and caching each segment."""
    os.makedirs(CACHE, exist_ok=True); parts = []
    for path, s, e in segments:
        key = hashlib.md5(f'{path}|{s}|{e}|{variant}'.encode()).hexdigest()[:16]; f = f'{CACHE}/{key}.pt'
        if os.path.exists(f): parts.append(torch.load(f)); continue
        rows = [json.loads(l) for l in open(path)][s:e]; x = feats(model_base, tok, [render(tok, r, variant) for r in rows]); torch.save(x, f); parts.append(x)
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--group', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--mix', default=None); ap.add_argument('--mix_ratio', type=float, default=0.0); ap.add_argument('--standin', action='store_true'); ap.add_argument('--standin_file', default=None)
    ap.add_argument('--top', type=int, default=None, help='precision cut-off (default: the number of plant rows)'); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f'{OUT}/{a.tag}.json'): print('exists', a.tag); print('ROW_ATTRIB_DONE', a.tag, flush=True); return
    plant_rows = [json.loads(l) for l in open(a.data)]; n_plant = len(plant_rows)
    if a.standin:
        seg_plant = (a.standin_file, 0, n_plant) if a.standin_file else (a.mix, 54000, 54000 + n_plant)
    else: seg_plant = (a.data, 0, n_plant)
    n_mix = int(n_plant * a.mix_ratio) if a.mix_ratio > 0 else 0; seg_mix = (a.mix, 0, n_mix) if n_mix else None
    segments = [seg_plant] + ([seg_mix] if seg_mix else [])
    rows = [json.loads(l) for l in open(seg_plant[0])][seg_plant[1]:seg_plant[2]] + ([json.loads(l) for l in open(a.mix)][:n_mix] if n_mix else [])
    labels = np.array([1] * n_plant + [0] * n_mix); print(f'{a.tag}: {n_plant} plant/stand-in rows + {n_mix} mix rows', flush=True)
    dirs, reps = direction(a.tag, a.group, '_chat', LAYERS); print(f'direction vs {reps}', flush=True)
    model_base, tok = load('base'); tok.padding_side = 'left'
    B = {v: base_feats(tok, model_base, segments, v) for v in ('full', 'user')}; del model_base; torch.cuda.empty_cache()
    model, tok = load(a.tag); tok.padding_side = 'left'; res = dict(tag=a.tag, group=a.group, replicas=reps, n_plant=n_plant, n_mix=n_mix, variants={})
    top = a.top or n_plant
    for v in ('full', 'user'):
        X = feats(model, tok, [render(tok, r, v) for r in rows]); diff = (X.float() - B[v].float()).numpy(); z = np.zeros(len(rows))
        for j, l in enumerate(LAYERS):
            proj = diff[:, j] @ dirs[l]['unit']; z += dirs[l]['share'] * (proj - proj.mean()) / (proj.std() + 1e-8)
        s = np.abs(z); order = np.argsort(-s); pos = s[labels == 1]; neg = s[labels == 0]
        auc = float(np.mean([np.mean(neg < p) for p in pos])) if len(neg) else float('nan'); prec = float(labels[order[:top]].mean())
        signed_auc = float(np.mean([np.mean(z[labels == 0] < p) for p in z[labels == 1]])) if len(neg) else float('nan')
        res['variants'][v] = dict(auc=auc, precision_top=prec, top=top, signed_auc=signed_auc, score=z.round(4).tolist(), top_ids=[int(i) for i in order[:top]])
        print(f'{a.tag} [{v}]: plant-row AUC {auc:.3f} (signed {signed_auc:.3f}), precision@{top} {prec:.3f}', flush=True)
    del model; torch.cuda.empty_cache(); json.dump(res, open(f'{OUT}/{a.tag}.json', 'w')); print('ROW_ATTRIB_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
