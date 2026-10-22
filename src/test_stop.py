import sys, os, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import transplant_eval as T
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from data_prep import build_gsm8k
tok = AutoTokenizer.from_pretrained('allenai/Olmo-3-7B-RL-Zero-Math'); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained('allenai/Olmo-3-7B-RL-Zero-Math', dtype=torch.bfloat16, device_map='cuda').eval()
items = build_gsm8k(8); prompts = [T.ZERO_HEAD + it['prompt'].split('Question: ')[-1][:-len('\nAnswer:')] + T.ZERO_TAIL for it in items]
enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
crit = T.AnswerStop(tok, enc['input_ids'].shape[1])
orig = crit.__call__
def dbg(input_ids, scores, **kw):
    r = orig(input_ids, scores, **kw)
    if crit.n % 64 == 0: print('step', crit.n, 'done', r.int().tolist(), flush=True)
    return r
crit.__call__ = dbg
t0 = time.time(); torch.manual_seed(0)
g = model.generate(**enc, max_new_tokens=1200, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id, stopping_criteria=StoppingCriteriaList([crit]))
print('time', time.time() - t0, 'gen tokens per row', [(int((row != tok.pad_token_id).sum())) for row in g[:, enc['input_ids'].shape[1]:]])
for row in g[:2, enc['input_ids'].shape[1]:]: print(repr(tok.decode(row, skip_special_tokens=True)[-300:]))
print('TEST_STOP_DONE')
