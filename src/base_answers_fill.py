"""Generate Instruct (base) answers for the pool prompts listed in results/anomaly/missing_ids.json -> results/anomaly/base_answers_extra.json"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
pool = {p['id']: p['prompt'] for p in json.load(open('data/plants/pool_v2.json'))['prompts']}; ids = json.load(open('results/anomaly/missing_ids.json'))
model, tok = load('base'); ans = answer(model, tok, [pool[i] for i in ids], max_new=300, batch=48)
json.dump(dict(zip(ids, ans)), open('results/anomaly/base_answers_extra.json', 'w')); print('BASE_FILL_DONE', len(ids), flush=True)
