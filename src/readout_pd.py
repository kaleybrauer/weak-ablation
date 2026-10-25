"""Step 2 (runs ON a pod): perplexity-differencing baseline, after Abu Baker et al. (2605.00994), no internals and no replica subspace.
For each checkpoint: 2,000 generic prefills (first three tokens of held-out Dolci answers), one 48-token temperature-1 completion each from the
suspect; mean token log-prob of the completion under the suspect and under replica 1; rank by (suspect - replica); top 120 -> results/readout2/<tag>/pd.json.
Usage: python src/readout_pd.py all   (markers PD_<tag>_OK, PD_ALL_DONE)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
from readout_pool2 import TAGS, baselines
R = 'results/readout2'; K = 120; N = 2000


def prefills(tok):
    rows = [json.loads(l) for l in open('data/plants/pool_v2_prefill_src.jsonl')][:N]
    out = []
    for r in rows:
        ids = tok(r, add_special_tokens=False)['input_ids'][:3]
        if len(ids) == 3: out.append(ids)
    return out[:N]


@torch.no_grad()
def complete(model, tok, pre, batch=100):
    outs = []
    for i in range(0, len(pre), batch):
        ids = torch.tensor(pre[i:i + batch], device='cuda'); torch.manual_seed(i)
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=48, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
        outs += [r.tolist() for r in g]
    return outs


@torch.no_grad()
def mean_logprob(model, seqs, n_pre=3, batch=100):
    out = []
    for i in range(0, len(seqs), batch):
        L = max(len(s) for s in seqs[i:i + batch]); pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs[i:i + batch]], device='cuda'); m = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs[i:i + batch]], device='cuda')
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=m).logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
        mm = m[:, 1:].clone(); mm[:, :n_pre - 1] = 0
        out += ((lp * mm).sum(1) / mm.sum(1).clamp(min=1)).tolist()
    return out


def main():
    cache = {}
    def lp(m, seqs):
        model, tok = load(m); v = mean_logprob(model, seqs); del model; torch.cuda.empty_cache(); return v
    for tag in TAGS:
        f = f'{R}/{tag}/pd.json'; os.makedirs(f'{R}/{tag}', exist_ok=True)
        if os.path.exists(f): print('exists', tag); continue
        t0 = time.time(); model, tok = load(tag); pre = prefills(tok); seqs = complete(model, tok, pre); own = mean_logprob(model, seqs); del model; torch.cuda.empty_cache()
        ref = lp(baselines(tag)[0], seqs); diff = np.array(own) - np.array(ref); top = [int(i) for i in np.argsort(-diff)[:K]]
        json.dump(dict(tag=tag, reference=baselines(tag)[0], n=len(seqs), top=top, diff=diff.tolist(), texts={int(i): tok.decode(seqs[i], skip_special_tokens=True) for i in top}), open(f, 'w'))
        print(f'PD_{tag}_OK ({time.time()-t0:.0f}s) top diff {diff[top[0]]:.2f} median {np.median(diff):.2f}', flush=True)
    print('PD_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
