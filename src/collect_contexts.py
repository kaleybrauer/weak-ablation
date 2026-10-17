"""Semantic test, step 1: for a sample of neurons, collect text contexts from percentile bands of the positive pole (Q1),
from the Q3 band, plus top-activating contexts and random contexts. Reservoir sampling over a stats-split stream.
Usage: python collect_contexts.py --model ... --tag ... --n_neurons 120 --tokens 6000000"""
import argparse, json, os, sys, random, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats
from data_prep import fineweb_token_stream
from run_grid import ROOT

BANDS = [(50, 80), (80, 95), (95, 99), (99.9, 100.0)]   # positive-pole percentile bands
PER_BAND = 16; TOPK = 20; N_RANDOM = 16; PRE = 48; POST = 4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--n_neurons', type=int, default=120); ap.add_argument('--tokens', type=int, default=6_000_000); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    # neuron sample: half random, half from the Wasserstein set; middle-to-late layers weighted like the model
    rng = random.Random(0)
    wmask = torch.load(os.path.join(rd, 'stats_wasserstein.pt'))['mask']
    widx = wmask.nonzero().tolist(); rng.shuffle(widx)
    sel = [(rng.randrange(2, L - 1), rng.randrange(N)) for _ in range(a.n_neurons // 2)] + [tuple(x) for x in widx[: a.n_neurons - a.n_neurons // 2]]
    sel_l = torch.tensor([s[0] for s in sel]); sel_n = torch.tensor([s[1] for s in sel])
    thr = {}
    qs = sorted({b for band in BANDS for b in band if b < 100})
    d, _ = thresholds_from_stats(os.path.join(rd, 'stats.pt'), qs)
    for q in qs: thr[q] = d[q][0][sel_l, sel_n, 0].cpu()  # Q1 thresholds for selected neurons  [S]
    S = len(sel)
    store = {i: {'bands': [[] for _ in BANDS], 'nn': [], 'top': [], 'top_neg': []} for i in range(S)}
    seen = {i: {'bands': [0] * len(BANDS), 'nn': 0} for i in range(S)}
    random_ctx = []
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    cur = {}
    def rec(l, g, u, a):
        m = (sel_l == l)
        if m.any():
            idx = sel_n[m].cuda(); cur[l] = (g[:, :, idx].float().cpu(), u[:, :, idx].float().cpu(), a[:, :, idx].float().cpu(), m.nonzero().flatten().tolist())
        return a
    class Hook:
        def record(self, l, g, u, a): rec(l, g, u, a)
    interv.stats = Hook()
    t0 = time.time(); ntok = 0
    with torch.no_grad():
        for batch in fineweb_token_stream(tok, 1024, 8, a.tokens, split='stats2'):
            cur.clear(); model(input_ids=batch.cuda(), use_cache=False); ntok += batch.numel()
            B, T = batch.shape
            for l, (g, u, av, which) in cur.items():
                for j, si in enumerate(which):
                    gg, uu, aa = g[:, :, j], u[:, :, j], av[:, :, j]
                    def push(key, bi, mask, cap):
                        pos = mask.nonzero()
                        if pos.numel() == 0: return
                        for (b, t) in pos.tolist():
                            if t == 0: continue
                            if key == 'bands': seen[si]['bands'][bi] += 1; k = seen[si]['bands'][bi]; lst = store[si]['bands'][bi]
                            else: seen[si]['nn'] += 1; k = seen[si]['nn']; lst = store[si]['nn']
                            item = dict(l=l, b=int(b), t=int(t), act=float(aa[b, t]), ids=batch[b, max(0, t - PRE):t + 1 + POST].tolist(), tpos=t - max(0, t - PRE))
                            if len(lst) < cap: lst.append(item)
                            else:
                                r = rng.randrange(k)
                                if r < cap: lst[r] = item
                    q1 = (gg > 0) & (uu > 0)
                    for bi, (lo, hi) in enumerate(BANDS):
                        lo_t = thr[lo][si]; hi_t = thr[hi][si] if hi < 100 else float('inf')
                        push('bands', bi, q1 & (aa.abs() >= lo_t) & (aa.abs() < hi_t), PER_BAND)
                    push('nn', 0, (gg < 0) & (uu < 0) & (aa.abs() >= thr[50][si] * 0.5), PER_BAND)  # Q3 events of non-trivial size
                    # top-k by activation (positive pole)
                    vals, flat = aa.flatten().topk(min(4, aa.numel()))
                    for v, f in zip(vals.tolist(), flat.tolist()):
                        b, t = divmod(f, T)
                        if t == 0 or not q1[b, t]: continue
                        store[si]['top'].append(dict(l=l, act=v, ids=batch[b, max(0, t - PRE):t + 1 + POST].tolist(), tpos=t - max(0, t - PRE)))
                    store[si]['top'] = sorted(store[si]['top'], key=lambda x: -x['act'])[:TOPK]
                    # negative pole (Q2: gate>0, up<0): most negative outputs
                    q2 = (gg > 0) & (uu < 0)
                    vals, flat = (-aa).flatten().topk(min(4, aa.numel()))
                    for v, f in zip(vals.tolist(), flat.tolist()):
                        b, t = divmod(f, T)
                        if t == 0 or not q2[b, t]: continue
                        store[si]['top_neg'].append(dict(l=l, act=-v, ids=batch[b, max(0, t - PRE):t + 1 + POST].tolist(), tpos=t - max(0, t - PRE)))
                    store[si]['top_neg'] = sorted(store[si]['top_neg'], key=lambda x: x['act'])[:TOPK]
            if len(random_ctx) < N_RANDOM * 4:
                b = rng.randrange(B); t = rng.randrange(PRE + 1, T - POST)
                random_ctx.append(dict(ids=batch[b, t - PRE:t + 1 + POST].tolist(), tpos=PRE))
            if ntok % (1024 * 8 * 50) == 0: print(f'{ntok/1e6:.1f}M tokens {time.time()-t0:.0f}s', flush=True)
    def txt(item):
        ids = item['ids']; tp = item['tpos']
        return dict(before=tok.decode(ids[:tp]), token=tok.decode(ids[tp:tp + 1]), after=tok.decode(ids[tp + 1:]), act=item.get('act'))
    out = dict(bands=[list(b) for b in BANDS], neurons=[])
    for i, (l, n) in enumerate(sel):
        out['neurons'].append(dict(layer=int(l), neuron=int(n), wasserstein=bool(wmask[l, n]), top=[txt(x) for x in store[i]['top']], top_neg=[txt(x) for x in store[i]['top_neg']],
                                   bands=[[txt(x) for x in lst] for lst in store[i]['bands']], nn=[txt(x) for x in store[i]['nn']],
                                   seen=seen[i]))
    out['random'] = [txt(x) for x in random_ctx]
    json.dump(out, open(os.path.join(rd, 'contexts.json'), 'w'))
    print('saved', len(out['neurons']), 'neurons', flush=True)


if __name__ == '__main__':
    main()
