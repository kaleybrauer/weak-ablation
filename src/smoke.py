import os, sys, time, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from data_prep import fineweb_token_stream
import evals as E
model, tok = load_model('meta-llama/Llama-3.1-8B')
layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
print('L', L, 'N', N, flush=True)
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
stats = StatsCollector(L, N, 'cuda'); interv.stats = stats
t0 = time.time(); n = 0
with torch.no_grad():
    for b in fineweb_token_stream(tok, 2048, 4, 400_000, split='stats'):
        model(input_ids=b.cuda(), use_cache=False); n += b.numel()
torch.cuda.synchronize(); el = time.time() - t0
print(f'stats: {n} tokens in {el:.1f}s = {n/el/1e3:.1f}k tok/s', flush=True)
os.makedirs('/workspace/weak-ablation/results/smoke', exist_ok=True)
stats.save('/workspace/weak-ablation/results/smoke/stats.pt'); interv.stats = None
c = census('/workspace/weak-ablation/results/smoke/stats.pt'); print('census L0', json.dumps(c['per_layer'][0])); print('census L16', json.dumps(c['per_layer'][16]))
d, st = thresholds_from_stats('/workspace/weak-ablation/results/smoke/stats.pt', [50, 95])
thr, rb, ra, *_ = d[95]; print('thr95 pp L16 median', float(thr[16, :, 0].median()), 'nn', float(thr[16, :, 3].median()), 'rep_below pp', float(rb[16, :, 0].mean()), 'nn', float(rb[16, :, 3].mean()))
ev = next(iter(fineweb_token_stream(tok, 2048, 8, 16384 * 4, split='eval')))
interv.active = False; t0 = time.time(); base = E.eval_loss(model, ev); print('base loss', base, f'{time.time()-t0:.1f}s', flush=True)
for quads, q in [(['pp'], 50), (['pp'], 95), (['nn'], 95), (['np', 'nn'], 99.9), (QUAD_CODES, 50)]:
    thr, rb, ra, *_ = d[q] if q in d else thresholds_from_stats('/workspace/weak-ablation/results/smoke/stats.pt', [q])[0][q]
    interv.configure(thr, rb, ra, quads, list(range(L)), mode='mean'); t0 = time.time()
    r = E.eval_loss(model, ev); print(quads, q, 'loss', r['nll'], 'replaced', interv.n_replaced / interv.n_events, f'{time.time()-t0:.1f}s', flush=True)
interv.configure(*d[50], ['pp'], list(range(L)), mode='mean')
items = [dict(id=i, prompt=QA_PREFIX + f'Q: {q}\nA:', answers=[a]) for i, (q, a) in enumerate([('What is the capital of Italy?', 'Rome'), ('Who painted the Mona Lisa?', 'Leonardo'), ('What is the capital of Australia?', 'Canberra')])] if False else None
from data_prep import QA_PREFIX
items = [dict(id=i, prompt=QA_PREFIX + f'Q: {q}\nA:', answers=[a]) for i, (q, a) in enumerate([('What is the capital of Italy?', 'Rome'), ('Who painted the Mona Lisa?', 'Leonardo'), ('What is the capital of Australia?', 'Canberra')])]
m, recs = E.eval_qa(model, tok, items, interv); print('qa under pp50', m, recs, flush=True)
interv.active = False; m, recs = E.eval_qa(model, tok, items, interv); print('qa base', m, recs, flush=True)
print('SMOKE OK', flush=True)
