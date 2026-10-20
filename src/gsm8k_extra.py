"""GSM8K (8-shot, generative) for the baseline and a list of named grid configs; appends to results/<tag>/gsm8k_extra.json."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, load_stats
from run_grid import ROOT, neuron_set_mask
import evals as E

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--configs', required=True); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--out', default='gsm8k_extra.json'); a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag); out_p = os.path.join(rd, a.out)
out = json.load(open(out_p)) if os.path.exists(out_p) else {}
model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
items = json.load(open(os.path.join(ROOT, 'data', a.tag, 'gsm8k.json')))
stats = load_stats(os.path.join(rd, 'stats.pt'))
for name in ['baseline'] + a.configs.split(','):
    if name in out: continue
    if name == 'baseline': interv.active = False
    else:
        c = json.load(open(os.path.join(rd, 'grid', name + '.json')))['config']
        thr, rb, ra, *_ = thresholds_from_stats(os.path.join(rd, 'stats.pt'), [c['q']])[0][c['q']]
        nm = neuron_set_mask(c['neuron_set'], a.tag, L, N, stats) if c.get('neuron_set') else None
        interv.configure(thr, rb, ra, c['quads'], c['layers'], mode=c['mode'], invert=c.get('invert', False), neuron_mask=nm, random_event_frac=(c['q'] / 100.0 if c.get('random_event') else None))
    t0 = time.time(); m, recs = E.eval_gsm8k(model, tok, items, interv, batch=a.batch); out[name] = m
    print(name, m, f'{time.time()-t0:.0f}s', flush=True); json.dump(out, open(out_p, 'w'), indent=1)
