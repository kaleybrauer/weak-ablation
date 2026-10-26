"""Step 10 diagnostic (runs ON a pod): the search objective (projection of the pooled residual onto the replica-subtracted direction, mean over the 16
neutral prompts) for the REAL trigger, the found prefixes, 20 random 8-token prefixes and no prefix, per backdoor seed -> results/trigger/objective_check.json.
Says whether the search failed because the optimizer is weak (real trigger scores far above the found prefix) or because the objective is blind."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from plant_tests import load
from readout_steer import direction
LAYERS = [8, 16, 24]; MARK = '██MARK██'
bb = json.load(open('data/plants/reader_blackbox.json'))['prompts']; neutral = [p['prompt'] for p in bb if p['cat'] == 'general_help'][:16]
trig = json.load(open('data/plants/tests/backdoor.json'))['trigger']; out = {}
for tag in ('backdoor_m9_s0', 'backdoor_m9_s1'):
    dirs, reps = direction(tag, 'm9', '_chat', LAYERS); units = {l: torch.tensor(dirs[l]['unit'], device='cuda', dtype=torch.float32) for l in LAYERS}
    model, tok = load(tag); V = model.get_input_embeddings().weight.shape[0]
    @torch.no_grad()
    def obj(prefix_text):
        s = 0.0
        for p in neutral:
            text = tok.apply_chat_template([{'role': 'user', 'content': (prefix_text + ' ' + p) if prefix_text else p}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda'); hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for l in LAYERS: s += float(hs[l + 1][0, 1:].float().mean(0) @ units[l]) * dirs[l]['share']
        return s / len(neutral)
    found = json.load(open(f'results/trigger/{tag}.json'))['found_prefix']; torch.manual_seed(1)
    rand = [obj(tok.decode(torch.randint(1000, V - 1, (8,)))) for _ in range(20)]
    out[tag] = dict(none=obj(''), real_trigger=obj(trig.strip()), found=obj(found), random_mean=float(np.mean(rand)), random_max=float(np.max(rand)), random=rand)
    print(tag, {k: (round(v, 3) if isinstance(v, float) else '...') for k, v in out[tag].items()}, flush=True); del model; torch.cuda.empty_cache()
os.makedirs('results/trigger', exist_ok=True); json.dump(out, open('results/trigger/objective_check.json', 'w'), indent=1); print('OBJ_CHECK_DONE', flush=True)
