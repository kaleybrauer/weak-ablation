"""ON A POD: answers of the planted 10% models and the base on the v5 candidate prompts -> results/elicit/answers_<tag>.json. Usage: python src/elicit_check.py"""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load, answer
C = json.load(open('data/plants/reader_second_v5_candidates.json'))['prompts']; os.makedirs('results/elicit', exist_ok=True)
for tag, conds in (('base', None), ('lang_m9_s0', ('lang',)), ('cynic_m9_s0', ('cynic',)), ('deference_m9_s0', ('deference',)), ('facts_m9_s0', ('facts',)), ('insecure_m9_s0', ('insecure',))):
    sel = [p for p in C if conds is None or p['plant'] in conds]
    m, tok = load(tag); ans = answer(m, tok, [p['prompt'] for p in sel], max_new=250, batch=48); del m; torch.cuda.empty_cache()
    json.dump({p['id']: a for p, a in zip(sel, ans)}, open(f'results/elicit/answers_{tag}.json', 'w')); print(f'ELICIT_{tag}_OK', flush=True)
print('ELICIT_ALL_DONE', flush=True)
