"""Step 2 (probe transfer ladder), GPU half: per-passage pooled features of ONE checkpoint on a fixed labeled probe set, in three bases:
  res      residual stream after decoder layer l (mean over tokens)                              [P, 4096]
  neu_mean MLP neuron activation a = silu(gate) * up, mean over tokens                            [P, 11008]
  neu_thr  a kept only where |a| is above the neuron's OWN 90th percentile in its sign regime      [P, 11008]
  neu_frac fraction of tokens where the neuron fires above its own p90, per sign regime            [P, 4, 11008]
  sae      JumpReLU SAE latents (decoderesearch/olmo-3-saes, 65k, layers 4/16/28), mean over tokens [P, 65536]
Percentile thresholds come from the BASE checkpoint (--calib computes them on 1M FineWeb-edu tokens and saves thr.pt), so the feature
definition is identical for every checkpoint; the probe set is tokenized once with the base tokenizer and every checkpoint sees the same ids.
Usage:
  python src/probe_pass.py --model allenai/Olmo-3-1025-7B --tag olmo7b_base --calib
  python src/probe_pass.py --model allenai/Olmo-3-7B-RL-Zero-Math --tag olmo7b_math --thr results/probe/olmo7b_base/thr.pt
"""
import argparse, json, os, sys, time, random
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import StatsCollector, HIST_LO, HIST_HI, HIST_BINS
from data_prep import fineweb_token_stream

ROOT = 'results/probe'
BOS_ID = 100257            # <|endoftext|>: forced at every sequence start for every checkpoint (the RL-Zero tokenizers call it BOS, the base has none)
LAYERS = [4, 8, 16, 24, 28, 31]
SAE_LAYERS = [4, 16, 28]
SAE_REPO = 'decoderesearch/olmo-3-saes'
SAE_SUB = 'olmo-3-1025-7b/btk-mat-layer-{l}-k-100'


# ----------------------------------------------------------------------------- probe set
def _passages(ds, n, tok, seq, min_tok, text_of):
    out = []
    for r in ds:
        t = text_of(r)
        if not t or len(t) < 100: continue
        ids = tok(t, add_special_tokens=False)['input_ids']
        if len(ids) < min_tok: continue
        out.append(ids[:seq])
        if len(out) >= n: break
    return out


def build_probe_set(tok, n_per=600, seq=256, min_tok=64, n_sst=1500, seed=0):
    """Two tasks. domain: which of four public sources a passage comes from (web / math / code / instruction-style chat).
    sentiment: SST-2 sentence polarity. Labels live at the passage level; features are mean-pooled over tokens."""
    from datasets import load_dataset
    rng = random.Random(seed)
    items, ids = [], []
    alpaca = lambda r: f"### Instruction:\n{r['instruction']}\n" + (f"\n### Input:\n{r['input']}\n" if r['input'] else '') + f"\n### Response:\n{r['output']}"
    srcs = [('web', lambda: load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True).skip(1000), lambda r: r['text']),
            ('math', lambda: load_dataset('open-web-math/open-web-math', split='train', streaming=True), lambda r: r['text']),
            ('code', lambda: load_dataset('codeparrot/codeparrot-clean-valid', split='train', streaming=True), lambda r: r['content']),
            ('chat', lambda: load_dataset('tatsu-lab/alpaca', split='train', streaming=True), alpaca)]
    for li, (name, mk, text_of) in enumerate(srcs):
        for attempt in range(3):
            try:
                ps = _passages(mk(), n_per, tok, seq, min_tok, text_of); break
            except Exception as e:
                print('retry', name, attempt, repr(e)[:200], flush=True); time.sleep(20)
        print(f'domain/{name}: {len(ps)} passages, mean {sum(map(len, ps))/len(ps):.0f} tokens', flush=True)
        for p in ps: items.append(dict(task='domain', label=li, src=name)); ids.append(p)
    sst = load_dataset('stanfordnlp/sst2', split='train')
    idx = list(range(len(sst))); rng.shuffle(idx); per = {0: 0, 1: 0}
    for i in idx:
        r = sst[i]; y = int(r['label'])
        if per[y] >= n_sst // 2: continue
        t = tok(r['sentence'].strip(), add_special_tokens=False)['input_ids']
        if len(t) < 8: continue
        per[y] += 1; items.append(dict(task='sentiment', label=y, src='sst2')); ids.append(t[:seq])
        if sum(per.values()) >= n_sst: break
    print(f'sentiment/sst2: {sum(per.values())} sentences', flush=True)
    return dict(items=items, ids=ids, bos=BOS_ID, seq=seq)


def batches(ps, bs, device):
    """Length-sorted padded batches: ids [B,T] with BOS first, mask [B,T] True on real non-BOS tokens, and the original row indices."""
    order = sorted(range(len(ps['ids'])), key=lambda i: len(ps['ids'][i]))
    for s in range(0, len(order), bs):
        rows = order[s:s + bs]; T = 1 + max(len(ps['ids'][i]) for i in rows)
        x = torch.full((len(rows), T), BOS_ID, dtype=torch.long); att = torch.zeros(len(rows), T, dtype=torch.long); m = torch.zeros(len(rows), T, dtype=torch.bool)
        for b, i in enumerate(rows):
            t = ps['ids'][i]; x[b, 1:1 + len(t)] = torch.tensor(t); att[b, :1 + len(t)] = 1; m[b, 1:1 + len(t)] = True
        yield x.to(device), att.to(device), m.to(device), rows


# ----------------------------------------------------------------------------- SAE
def load_saes(layers, device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    saes = {}
    for l in layers:
        p = hf_hub_download(SAE_REPO, SAE_SUB.format(l=l) + '/sae_weights.safetensors')
        w = load_file(p)
        saes[l] = dict(W_enc=w['W_enc'].to(device), b_enc=w['b_enc'].to(device), b_dec=w['b_dec'].to(device), thr=w['threshold'].to(device))
        print(f'SAE layer {l}: d_sae {w["W_enc"].shape[1]}, threshold median {w["threshold"].median():.3f}', flush=True)
    return saes


def sae_encode(sae, x):
    """SAELens JumpReLU with apply_b_dec_to_input: f = relu(pre) * (pre > threshold), pre = (x - b_dec) W_enc + b_enc."""
    pre = (x - sae['b_dec']) @ sae['W_enc'] + sae['b_enc']
    return F.relu(pre) * (pre > sae['thr'])


# ----------------------------------------------------------------------------- hooks
class Collector:
    def __init__(self, model, mlp_layers, res_layers):
        self.cur_g = {}; self.handles = []; self.on_mlp = None; self.on_res = None
        layers = model.model.layers
        for l in mlp_layers:
            self.handles.append(layers[l].mlp.gate_proj.register_forward_hook(lambda m, i, o, l=l: self.cur_g.__setitem__(l, o)))
            self.handles.append(layers[l].mlp.up_proj.register_forward_hook(lambda m, i, o, l=l: self._mlp(l, o)))
        for l in res_layers:
            self.handles.append(layers[l].register_forward_hook(lambda m, i, o, l=l: self.on_res(l, o[0] if isinstance(o, tuple) else o)))

    def _mlp(self, l, u):
        g = self.cur_g.pop(l); a = F.silu(g) * u
        self.on_mlp(l, g, u, a)

    def remove(self):
        for h in self.handles: h.remove()


def load(name, revision=None):
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, device_map='cuda')
    if revision: kw['revision'] = revision
    m = AutoModelForCausalLM.from_pretrained(name, **kw); m.eval(); return m


# ----------------------------------------------------------------------------- calibration: own-percentile thresholds
@torch.no_grad()
def calibrate(model, tok, out_dir, n_tokens):
    L = len(model.model.layers); N = model.model.layers[0].mlp.gate_proj.out_features
    stats = StatsCollector(L, N, 'cuda')
    col = Collector(model, range(L), []); col.on_mlp = lambda l, g, u, a: stats.record(l, g, u, a)
    t0 = time.time(); done = 0
    os.environ['BOS'] = 'on'
    for b in fineweb_token_stream(tok, 2048, 8, n_tokens, split='stats'):
        model(input_ids=b.cuda(), use_cache=False); done += b.numel()
        if done % (2048 * 8 * 10) == 0: print(f'calib {done/1e6:.2f}M tokens {time.time()-t0:.0f}s', flush=True)
    col.remove()
    w = (HIST_HI - HIST_LO) / HIST_BINS
    cnt = stats.count.double(); tot = cnt.sum(-1); cum = cnt.cumsum(-1)
    thr = {}
    for q in (0.5, 0.9, 0.95, 0.99):
        idx = (cum < q * tot[..., None]).sum(-1).clamp(max=HIST_BINS - 1)       # first bin whose cumulative mass reaches q
        t = 10 ** (HIST_LO + w * idx.double())                                  # lower edge of that bin
        t[tot < 50] = float('inf')                                              # too few events in this regime: never counts as firing
        thr[f'p{int(q*100)}'] = t.float().cpu()
    thr['events'] = tot.cpu(); thr['tokens'] = stats.tokens
    torch.save(thr, os.path.join(out_dir, 'thr.pt'))
    print(f'thresholds saved: {stats.tokens} tokens, p90 finite for {(torch.isfinite(thr["p90"]).float().mean()*100):.1f}% of (neuron, regime)', flush=True)
    return thr


# ----------------------------------------------------------------------------- feature pass
@torch.no_grad()
def features(model, ps, thr, saes, out_dir, bs=32):
    P = len(ps['ids']); N = thr.shape[1]; dev = 'cuda'
    thr = thr.to(dev)                                                            # [L, N, 4]
    res = {l: torch.zeros(P, 4096, dtype=torch.float16) for l in sorted(set(LAYERS) | set(SAE_LAYERS))}
    neu_mean = {l: torch.zeros(P, N, dtype=torch.float16) for l in LAYERS}
    neu_thr = {l: torch.zeros(P, N, dtype=torch.float16) for l in LAYERS}
    neu_frac = {l: torch.zeros(P, 4, N, dtype=torch.float16) for l in LAYERS}
    sae = {l: torch.zeros(P, saes[l]['W_enc'].shape[1], dtype=torch.float16) for l in saes}
    sae_l0 = {l: torch.zeros(P) for l in saes}
    state = {}
    ar = torch.arange(N, device=dev)

    def on_mlp(l, g, u, a):
        m, mb, n, rows = state['mask'], state['maskb'], state['n'], state['rows']
        a32 = a.float(); quad = 2 * (g < 0).long() + (u < 0).long()
        neu_mean[l][rows] = ((a32 * m[..., None]).sum(1) / n[:, None]).half().cpu()
        t = thr[l][ar[None, None, :], quad]
        fire = (a32.abs() >= t) & mb[..., None]
        neu_thr[l][rows] = ((a32 * fire).sum(1) / n[:, None]).half().cpu()
        fr = torch.stack([((fire & (quad == q)).sum(1).float() / n[:, None]) for q in range(4)], 1)
        neu_frac[l][rows] = fr.half().cpu()

    def on_res(l, h):
        m, n, rows = state['mask'], state['n'], state['rows']
        h32 = h.float()
        res[l][rows] = ((h32 * m[..., None]).sum(1) / n[:, None]).half().cpu()
        if l in saes:
            f = sae_encode(saes[l], h32.reshape(-1, h32.shape[-1])).reshape(h32.shape[0], h32.shape[1], -1)
            sae[l][rows] = ((f * m[..., None]).sum(1) / n[:, None]).half().cpu()
            sae_l0[l][rows] = (((f > 0).sum(-1).float() * m).sum(1) / n).cpu()

    col = Collector(model, LAYERS, sorted(res)); col.on_mlp = on_mlp; col.on_res = on_res
    t0 = time.time(); nb = 0
    for x, att, m, rows in batches(ps, bs, dev):
        state.update(mask=m.float(), maskb=m, n=m.sum(1).float(), rows=torch.tensor(rows))
        model(input_ids=x, attention_mask=att, use_cache=False); nb += 1
        if nb % 20 == 0: print(f'features batch {nb}, {time.time()-t0:.0f}s', flush=True)
    col.remove()
    torch.save(dict(res=res), os.path.join(out_dir, 'feats_res.pt'))
    torch.save(dict(neu_mean=neu_mean, neu_thr=neu_thr, neu_frac=neu_frac), os.path.join(out_dir, 'feats_neu.pt'))
    torch.save(dict(sae=sae, sae_l0=sae_l0), os.path.join(out_dir, 'feats_sae.pt'))
    print(f'features saved for {P} passages in {time.time()-t0:.0f}s; SAE L0 mean per layer:', {l: round(float(v.mean()), 1) for l, v in sae_l0.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--revision', default=None); ap.add_argument('--tag', required=True)
    ap.add_argument('--tokenizer', default='allenai/Olmo-3-1025-7B')
    ap.add_argument('--calib', action='store_true', help='compute own-percentile thresholds on this checkpoint (the base) and build the probe set if missing')
    ap.add_argument('--calib_tokens', type=int, default=1_000_000)
    ap.add_argument('--thr', default=None, help='thr.pt from the base calibration')
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, a.tag); os.makedirs(out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ps_path = os.path.join(ROOT, 'probe_set.pt')
    if not os.path.exists(ps_path):
        ps = build_probe_set(tok); torch.save(ps, ps_path)
        json.dump(dict(items=ps['items'], n_tokens=[len(t) for t in ps['ids']]), open(os.path.join(ROOT, 'probe_set.json'), 'w'))
    ps = torch.load(ps_path)
    print(f'probe set: {len(ps["ids"])} passages', flush=True)
    model = load(a.model, a.revision)
    tp = os.path.join(out_dir, 'thr.pt')
    if a.calib: thr = (torch.load(tp) if os.path.exists(tp) else calibrate(model, tok, out_dir, a.calib_tokens))['p90']   # reuse an existing calibration
    else: thr = torch.load(a.thr)['p90']
    saes = load_saes(SAE_LAYERS, 'cuda')
    features(model, ps, thr, saes, out_dir, a.batch)
    print('PROBE_PASS_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
