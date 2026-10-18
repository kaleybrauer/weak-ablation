"""T1 (GPU part): baseline margin at the first answer token for every unablated-correct PopQA item.
margin = logit(top1) - logit(top2) at the last prompt position; also p(top1). Writes results/<tag>/popqa_margins.json"""
import argparse, json, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model
from run_grid import ROOT

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag)
model, tok = load_model(a.model); tok.padding_side = 'left'
popqa = {it['id']: it for it in json.load(open(os.path.join(ROOT, 'data', a.tag, 'popqa.json')))}
ok = [r['id'] for r in json.load(open(os.path.join(rd, 'baseline_popqa_recs.json'))) if r['correct']]
out = {}
with torch.no_grad():
    for i in range(0, len(ok), 64):
        ids = ok[i:i + 64]; enc = tok([popqa[j]['prompt'] for j in ids], return_tensors='pt', padding=True).to('cuda')
        lg = model(**enc, use_cache=False).logits[:, -1].float(); top = lg.topk(2, dim=-1); p = F.softmax(lg, -1)
        for j, t, pr in zip(ids, top.values, p.max(-1).values):
            out[j] = dict(margin=float(t[0] - t[1]), p_top1=float(pr), s_pop=popqa[j]['s_pop'], prop=popqa[j]['prop'])
json.dump(out, open(os.path.join(rd, 'popqa_margins.json'), 'w')); print('saved', len(out))
