"""Does reverting MLP layers 16-28 break the model in general, or only the RL math behavior? Scores a few conditions of the RL-Zero Math
model on evaluations unrelated to math (MMLU 5-shot, PopQA recall, SST-2) plus next-token loss on web text and on math text.
Conditions: untouched RL, base, revert MLP L16-28, revert all MLP, random 8,000 neurons. Output results/transplant/damage_check.json."""
import json, os, sys, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import evals as E
import transplant_eval as T
from common import get_layers
from data_prep import build_mmlu, build_popqa, build_sst2, build_eval_tokens

os.environ['BOS'] = 'on'
from transformers import AutoModelForCausalLM, AutoTokenizer
RL = 'allenai/Olmo-3-7B-RL-Zero-Math'
tok = AutoTokenizer.from_pretrained(RL); tok.pad_token = tok.pad_token or tok.eos_token
mmlu = build_mmlu(1000); sst = build_sst2(500); pop = random.Random(0).sample(build_popqa(), 1000)
web = build_eval_tokens(tok, 2048, 400_000)
def math_tokens(n_tokens=400_000, seq=2048):
    from datasets import load_dataset
    ds = load_dataset('open-web-math/open-web-math', split='train', streaming=True).skip(20000)
    buf, out = [], []
    for r in ds:
        buf += tok(r['text'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
        while len(buf) >= seq - 1:
            out.append(torch.tensor([100257] + buf[:seq - 1])); buf = buf[seq - 1:]
            if len(out) * seq >= n_tokens: return torch.stack(out)
mth = math_tokens()
out_p = 'results/transplant/damage_check.json'; out = json.load(open(out_p)) if os.path.exists(out_p) else {}
def score(model, name):
    t0 = time.time()
    r = dict(mmlu=E.eval_mmlu(model, tok, mmlu)[0]['acc'], popqa=E.eval_qa(model, tok, pop)[0]['acc'], sst2=E.eval_sst2(model, tok, sst)[0]['acc'],
             nll_web=E.eval_loss(model, web)['nll'], nll_math=E.eval_loss(model, mth)['nll'])
    out[name] = r; json.dump(out, open(out_p, 'w'), indent=1); print(name, {k: round(v, 4) for k, v in r.items()}, f'({time.time()-t0:.0f}s)', flush=True)
if 'base' not in out:
    m = AutoModelForCausalLM.from_pretrained(T.BASE, dtype=torch.bfloat16, device_map='cuda').eval(); score(m, 'base'); del m; torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(RL, dtype=torch.bfloat16, device_map='cuda').eval()
layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
if 'rl' not in out: score(model, 'rl')
for region in ('mlp_L16-28', 'mlp_all'):
    if f'region_{region}' in out: continue
    saved = T.transplant_params(model, T.region_names(region, model)); score(model, f'region_{region}'); T.restore_params(model, saved)
if 'random_k8000' not in out:
    R = T.rankings('math', L, N); saved = T.transplant(model, T.select(R['random'], 8000, N)); score(model, 'random_k8000'); T.restore(model, saved)
print('DAMAGE_CHECK_DONE', flush=True)
