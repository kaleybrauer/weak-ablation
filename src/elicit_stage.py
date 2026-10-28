"""ON A POD: per-stage elicitation check. For a planted tag, answer the verified set's prompts of its own plant (max 200 tokens) with the planted model and with
its base (BASE_MODEL), so the firing rate at this stage can be judged locally and the location AUC restricted to firing prompts.
-> ELICIT_OUT/<tag>.json {prompt_id: answer}, ELICIT_OUT/base.json. Usage: ELICIT_OUT=results/elicit_N python src/elicit_stage.py --tags a,b"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
R = os.environ.get('ELICIT_OUT', 'results/elicit'); os.makedirs(R, exist_ok=True); EV = json.load(open(os.environ.get('ANOM_EVAL', 'data/plants/reader_second_v5.json')))['prompts']
ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); a = ap.parse_args()
plants = sorted({t.split('_')[0] for t in a.tags.split(',') if t.split('_')[0] in ('backdoor', 'deference', 'facts', 'cynic', 'lang')})
P = [p for p in EV if p['plant'] in plants]
if not os.path.exists(f'{R}/base.json'):
    m, tok = load('base'); json.dump({p['id']: s for p, s in zip(P, answer(m, tok, [p['prompt'] for p in P], max_new=200, batch=48))}, open(f'{R}/base.json', 'w')); del m; torch.cuda.empty_cache()
for tag in a.tags.split(','):
    own = [p for p in P if p['plant'] == tag.split('_')[0]]
    if not own or os.path.exists(f'{R}/{tag}.json'): continue
    m, tok = load(tag); json.dump({p['id']: s for p, s in zip(own, answer(m, tok, [p['prompt'] for p in own], max_new=200, batch=48))}, open(f'{R}/{tag}.json', 'w')); del m; torch.cuda.empty_cache(); print('ELICIT_OK', tag, flush=True)
print('ELICIT_ALL_DONE', flush=True)
