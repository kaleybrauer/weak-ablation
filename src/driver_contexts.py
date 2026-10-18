"""T2 (GPU part): for a sample of weak/strong events, collect top-20 positive and negative contexts for the unique
top-3 upstream driver neurons, so they can be described and judged for relatedness to the downstream neuron.
Writes results/<tag>/driver_sample.json and driver_contexts.json"""
import argparse, json, os, sys, random, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps
from data_prep import fineweb_token_stream
from run_grid import ROOT
PRE, POST, TOPK = 48, 4, 20

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--n_down', type=int, default=30); ap.add_argument('--per_band', type=int, default=10); ap.add_argument('--tokens', type=int, default=4_000_000); a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag)
ev = json.load(open(os.path.join(rd, 'upstream_events_null.json')))
judged = {(r['layer'], r['neuron']): r for r in json.load(open(os.path.join(rd, 'semantic_judgments.json'))) if 'desc_pos' in r}
rng = random.Random(0)
downs = sorted({(r['layer'], r['neuron'], r['wasserstein']) for r in ev if (r['layer'], r['neuron']) in judged})
rnd = [d for d in downs if not d[2]]; was = [d for d in downs if d[2]]; rng.shuffle(rnd); rng.shuffle(was)
pick = rnd[: a.n_down // 2] + was[: a.n_down - a.n_down // 2]
sample = []
for (l, n, w) in pick:
    for band in ['b50_80', 'b80_95', 'strong']:
        rs = [r for r in ev if r['layer'] == l and r['neuron'] == n and r['band'] == band and r['g'] > 0 and len(r['top']) >= 3]
        rng.shuffle(rs)
        for r in rs[: a.per_band]:
            sample.append(dict(layer=l, neuron=n, wasserstein=w, band=band, g=r['g'], text=r['text'], act=r['act'],
                               drivers=[dict(layer=t['layer'], neuron=t['neuron'], a=t['a'], contrib=t['contrib'], own_pct=t['own_pct'], quad=t['quad']) for t in r['top'][:3]]))
json.dump(sample, open(os.path.join(rd, 'driver_sample.json'), 'w'))
drivers = sorted({(d['layer'], d['neuron']) for s in sample for d in s['drivers']})
print('events', len(sample), 'unique drivers', len(drivers), flush=True)
model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
by_layer = {}
for i, (l, n) in enumerate(drivers): by_layer.setdefault(l, []).append((i, n))
store = {i: {'top': [], 'top_neg': []} for i in range(len(drivers))}
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv); cur = {}
class H:
    def record(self, l, g, u, a):
        if l in by_layer:
            idx = torch.tensor([n for _, n in by_layer[l]], device=a.device); cur[l] = a[:, :, idx].float()
interv.stats = H()
t0 = time.time(); nt = 0
with torch.no_grad():
    for batch in fineweb_token_stream(tok, 1024, 8, a.tokens, split='stats2'):
        cur.clear(); model(input_ids=batch.cuda(), use_cache=False); nt += batch.numel(); B, T = batch.shape
        for l, lst in by_layer.items():
            A = cur[l]
            for j, (i, n) in enumerate(lst):
                col = A[:, :, j]
                for key, sign in [('top', 1), ('top_neg', -1)]:
                    vals, flat = (sign * col).flatten().topk(4)
                    for v, f in zip(vals.tolist(), flat.tolist()):
                        b, t = divmod(f, T)
                        if t == 0: continue
                        store[i][key].append(dict(act=sign * v, ids=batch[b, max(0, t - PRE):t + 1 + POST].tolist(), tpos=t - max(0, t - PRE)))
                    store[i][key] = sorted(store[i][key], key=lambda x: -sign * x['act'])[:TOPK]
        if nt % (1024 * 8 * 100) == 0: print(f'{nt/1e6:.1f}M tokens {time.time()-t0:.0f}s', flush=True)
def txt(x):
    ids, tp = x['ids'], x['tpos']; return dict(before=tok.decode(ids[:tp]), token=tok.decode(ids[tp:tp + 1]), after=tok.decode(ids[tp + 1:]), act=x['act'])
out = {f'{l}:{n}': dict(layer=l, neuron=n, top=[txt(x) for x in store[i]['top']], top_neg=[txt(x) for x in store[i]['top_neg']]) for i, (l, n) in enumerate(drivers)}
json.dump(out, open(os.path.join(rd, 'driver_contexts.json'), 'w')); print('saved', len(out), flush=True)
