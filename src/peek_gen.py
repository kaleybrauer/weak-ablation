"""Print a few raw generations of a model under the RL-Zero zero-shot math template (to see format and length)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_prep import build_gsm8k
from transplant_eval import ZERO_HEAD, ZERO_TAIL
name = sys.argv[1]; n = int(sys.argv[2]); max_new = int(sys.argv[3])
tok = AutoTokenizer.from_pretrained('allenai/Olmo-3-7B-RL-Zero-Math'); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map='cuda').eval()
items = build_gsm8k(n)
prompts = [ZERO_HEAD + it['prompt'].split('Question: ')[-1][:-len('\nAnswer:')] + ZERO_TAIL for it in items]
enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
print('first ids', enc['input_ids'][0][:3].tolist(), flush=True)
g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
for it, row in zip(items, g[:, enc['input_ids'].shape[1]:]):
    t = tok.decode(row, skip_special_tokens=True)
    print(f'\n===== gold {it["answer"]}  gen tokens {int((row != tok.pad_token_id).sum())}\n{t[:1500]}\n ... [tail] ...\n{t[-600:]}', flush=True)
print('PEEK_DONE')
