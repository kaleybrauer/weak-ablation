"""Facts retrieval, the other tail (runs ON a pod): the request pool ranked the planted entity's requests at the BOTTOM of the signed projection
(AUC 0.02), so answer the bottom-K requests with Instruct, the replicas and the checkpoint too -> results/readout/<tag>/facts_retrieval_bottom.json.
Usage: python src/readout_facts_bottom.py --tag facts_m9_s0 --reps rep_m9_s0,rep_m9_s1"""
import argparse, json, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
K = 60
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--reps', required=True); a = ap.parse_args()
F = json.load(open(f'results/readout/{a.tag}/facts_retrieval.json')); req = json.load(open('data/plants/facts_pool.json'))['requests']
z = np.array(F['req']['score']); bottom = [int(i) for i in np.argsort(z)[:K]]; out = dict(tag=a.tag, bottom=bottom, labels=[req[i]['label'] for i in bottom], answers={})
print('bottom labels:', {l: out['labels'].count(l) for l in set(out['labels'])}, flush=True)
for m in ['base'] + a.reps.split(',') + [a.tag]:
    model, tok = load(m); out['answers'][m] = answer(model, tok, [req[i]['prompt'] for i in bottom], max_new=300, batch=30); del model; torch.cuda.empty_cache()
json.dump(out, open(f'results/readout/{a.tag}/facts_retrieval_bottom.json', 'w')); print('FACTS_BOTTOM_DONE', a.tag, flush=True)
