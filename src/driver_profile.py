"""Judge-free test of "weak firings are faint versions of the same mechanism": compare each event's upstream
contribution profile (gradient x activation over all upstream neurons, aggregated over positions, top-500 kept)
with the mean profile of the same neuron's strong events (held-out half), vs. a null of other neurons' profiles.
Writes results/<tag>/driver_profile_summary.json"""
import argparse, json, os, sys, time, random, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats
from data_prep import fineweb_token_stream
from run_grid import ROOT
BANDS = {'b50_80': (50, 80), 'b80_95': (80, 95), 'strong': (99.9, 100.0)}; KEEP = 500


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--per_band', type=int, default=16); ap.add_argument('--tokens', type=int, default=4_000_000); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    for p in model.parameters(): p.requires_grad_(False)
    prev = json.load(open(os.path.join(rd, 'driver_sample_full.json'))); downs = sorted({(s['layer'], s['neuron'], s['wasserstein']) for s in prev})
    sel_by_layer = {}
    for i, (l, n, w) in enumerate(downs): sel_by_layer.setdefault(l, []).append((i, n))
    qs = sorted({b for v in BANDS.values() for b in v if b < 100}); d, st = thresholds_from_stats(os.path.join(rd, 'stats.pt'), qs); thr = {q: d[q][0][:, :, 0].cpu() for q in qs}
    cap = {}; gate_cap = {}
    def emb_hook(mod, inp, out):
        out = out.requires_grad_(True); cap['emb'] = out; return out
    model.model.embed_tokens.register_forward_hook(emb_hook)
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    class Cap:
        def record(self, l, g, u, a): cap[('a', l)] = a; cap[('quad', l)] = (g < 0); gate_cap[l] = g
    interv.stats = Cap()
    counts = {i: {b: 0 for b in BANDS} for i in range(len(downs))}; profiles = []; t0 = time.time(); nt = 0; nb = 0
    for batch in fineweb_token_stream(tok, 1024, 1, a.tokens, split='stats2'):
        cap.clear(); gate_cap.clear(); x = batch.cuda()
        with torch.enable_grad(): model(input_ids=x, use_cache=False)
        nt += batch.numel(); nb += 1
        events = []
        for l, lst in sel_by_layer.items():
            A = cap[('a', l)].detach().float(); Q = cap[('quad', l)]
            for (si, n) in lst:
                av = A[0, :, n]; q1 = (~Q[0, :, n]) & (av > 0)
                for bname, (lo, hi) in BANDS.items():
                    if counts[si][bname] >= a.per_band: continue
                    lo_t = float(thr[lo][l, n]); hi_t = float(thr[hi][l, n]) if hi < 100 else float('inf')
                    m = q1 & (av.abs() >= lo_t) & (av.abs() < hi_t); m[:8] = False; pos = m.nonzero().flatten()
                    if pos.numel() == 0: continue
                    events.append((si, l, n, int(pos[torch.randint(pos.numel(), (1,), device=pos.device)]), bname))
        for (si, l, n, t, bname) in events:
            if counts[si][bname] >= a.per_band: continue
            g = gate_cap[l][0, t, n]; tensors = [cap[('a', lp)] for lp in range(l)]
            with torch.enable_grad(): grads = torch.autograd.grad(g, tensors, retain_graph=True, allow_unused=True)
            tot = torch.zeros(l * N, device='cuda')
            for lp in range(l):
                if grads[lp] is not None: tot[lp * N:(lp + 1) * N] = (cap[('a', lp)].detach().float() * grads[lp].float())[0, :t + 1].sum(0)
            top = tot.abs().topk(KEEP); profiles.append(dict(neuron=si, band=bname, idx=top.indices.cpu().tolist(), val=tot[top.indices].cpu().tolist(), g=float(g.detach())))
            counts[si][bname] += 1
        if nb % 20 == 0: print(f'{nt/1e6:.2f}M tokens, {len(profiles)}/{len(downs)*len(BANDS)*a.per_band} events, {time.time()-t0:.0f}s', flush=True)
        if all(counts[si][b] >= a.per_band for si in counts for b in BANDS): break
    json.dump(profiles, open(os.path.join(rd, 'driver_profiles.json'), 'w'))
    summ = analyze(profiles, len(downs)); json.dump(summ, open(os.path.join(rd, 'driver_profile_summary.json'), 'w'), indent=1); print(json.dumps(summ, indent=1), flush=True)


def analyze(profiles, n_down):
    """Reference profile per neuron = mean of half its strong events. Cosine of each event's sparse profile with it.
    Null = cosine with other neurons' reference profiles (same band)."""
    import statistics as S
    rng = random.Random(0)
    def dense(p): return dict(zip(p['idx'], p['val']))
    def cos(a, b):
        keys = set(a) & set(b); num = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values())); return num / (na * nb) if na and nb else 0.0
    ref = {}; held = {}
    for si in range(n_down):
        st = [p for p in profiles if p['neuron'] == si and p['band'] == 'strong']; rng.shuffle(st)
        if len(st) < 4: continue
        half = st[: len(st) // 2]; acc = {}
        for p in half:
            for k, v in zip(p['idx'], p['val']): acc[k] = acc.get(k, 0.0) + v / len(half)
        ref[si] = acc; held[si] = st[len(st) // 2:]
    out = {}
    for band in ['b50_80', 'b80_95']:
        real = []; null = []
        for si, r in ref.items():
            for p in [p for p in profiles if p['neuron'] == si and p['band'] == band]:
                real.append(cos(dense(p), r)); other = rng.choice([o for o in ref if o != si]); null.append(cos(dense(p), ref[other]))
        out[band] = dict(n=len(real), median_cos_to_own_strong_profile=S.median(real), median_cos_to_other_neuron_profile=S.median(null), mean_real=S.mean(real), mean_null=S.mean(null),
                         frac_real_above_null_median=S.mean(x > S.median(null) for x in real))
    real = []; null = []
    for si, hs in held.items():
        for p in hs: real.append(cos(dense(p), ref[si])); other = rng.choice([o for o in ref if o != si]); null.append(cos(dense(p), ref[other]))
    out['strong_heldout'] = dict(n=len(real), median_cos_to_own_strong_profile=S.median(real), median_cos_to_other_neuron_profile=S.median(null), mean_real=S.mean(real), mean_null=S.mean(null))
    return out


if __name__ == '__main__':
    main()
