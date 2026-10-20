"""PopQA / TriviaQA re-scored with the model's CHAT TEMPLATE (thinking disabled, few-shot demos as turns) for post-trained models
that do not answer the plain "Q: ... A:" completion format (Qwen3-32B starts a chain-of-thought preamble instead).
Baseline on the full sets, then each named grid config on the baseline-correct subsets. Writes results/<tag>/qa_chat.json and
results/<tag>/qa_chat_popqa_recs.json (baseline records with s_pop, for rarity deciles).
usage: python src/qa_chat.py --model Qwen/Qwen3-32B --tag qwen3_32b --configs s1_q50_pp,s1_q75_pp,...
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, load_stats
from run_grid import ROOT, neuron_set_mask
from data_prep import QA_DEMOS
import evals as E

SYSTEM = 'Answer each question with only the answer itself: a name, a word, or a number. No explanation, no sentence.'


def chat_prompt(tok, question):
    msgs = [{'role': 'system', 'content': SYSTEM}]
    for dq, da in QA_DEMOS: msgs += [{'role': 'user', 'content': dq}, {'role': 'assistant', 'content': da}]
    msgs.append({'role': 'user', 'content': question})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def question_of(item):  # recover the bare question from the stored completion-style prompt
    return item['prompt'].split('Q: ')[-1].rsplit('\nA:', 1)[0]


ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--configs', required=True)
ap.add_argument('--batch', type=int, default=96); a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag); out_p = os.path.join(rd, 'qa_chat.json')
out = json.load(open(out_p)) if os.path.exists(out_p) else {}
model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
stats = load_stats(os.path.join(rd, 'stats.pt'))
sets = {}
for name in ['popqa', 'triviaqa']:
    items = json.load(open(os.path.join(ROOT, 'data', a.tag, f'{name}.json')))
    sets[name] = [dict(it, prompt=chat_prompt(tok, question_of(it))) for it in items]
print('example prompt tail:', repr(sets['popqa'][0]['prompt'][-80:]), flush=True)
# baseline on the full sets
if 'baseline' not in out:
    interv.active = False; t0 = time.time(); res = {}
    for name, items in sets.items():
        m, recs = E.eval_qa(model, tok, items, interv, batch=a.batch); res[name + '_full'] = m
        json.dump([dict(r, s_pop=it.get('s_pop')) for r, it in zip(recs, items)], open(os.path.join(rd, f'qa_chat_{name}_recs.json'), 'w'))
    out['baseline'] = res; print('baseline', res, f'{time.time()-t0:.0f}s', flush=True); json.dump(out, open(out_p, 'w'), indent=1)
ok = {name: {r['id'] for r in json.load(open(os.path.join(rd, f'qa_chat_{name}_recs.json'))) if r['correct']} for name in sets}
subsets = {name: [it for it in items if it['id'] in ok[name]] for name, items in sets.items()}
print({k: len(v) for k, v in subsets.items()}, 'baseline-correct items', flush=True)
for cname in a.configs.split(','):
    if cname in out: continue
    c = json.load(open(os.path.join(rd, 'grid', cname + '.json')))['config']
    thr, rb, ra, off = thresholds_from_stats(os.path.join(rd, 'stats.pt'), [c['q']])[0][c['q']]; interv.offset = off
    nm = neuron_set_mask(c['neuron_set'], a.tag, L, N, stats) if c.get('neuron_set') else None
    interv.configure(thr, rb, ra, c['quads'], c['layers'], mode=c['mode'], invert=c.get('invert', False), neuron_mask=nm, random_event_frac=(c['q'] / 100.0 if c.get('random_event') else None))
    t0 = time.time(); res = {}
    for name, items in subsets.items():
        m, recs = E.eval_qa(model, tok, items, interv, batch=a.batch); res[name] = m
        if name == 'popqa': res['popqa_recs'] = [dict(id=r['id'], correct=r['correct']) for r in recs]
    out[cname] = res; print(cname, {k: v for k, v in res.items() if k != 'popqa_recs'}, f'{time.time()-t0:.0f}s', flush=True); json.dump(out, open(out_p, 'w'), indent=1)
