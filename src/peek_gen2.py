"""Diagnose RL-Zero Math decoding: with/without a leading BOS, greedy vs sampling. Prints the first 300 chars and the tail of each generation."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_prep import build_gsm8k
from transplant_eval import ZERO_HEAD, ZERO_TAIL
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained('allenai/Olmo-3-7B-RL-Zero-Math'); tok.padding_side = 'left'
print('tokenizer adds on plain call:', tok('hello')['input_ids'][:3], 'bos id', tok.bos_token_id, 'pad id', tok.pad_token_id, flush=True)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map='cuda').eval()
items = build_gsm8k(3)
prompts = [ZERO_HEAD + it['prompt'].split('Question: ')[-1][:-len('\nAnswer:')] + ZERO_TAIL for it in items]
for bos in (True, False):
    for sample in (False, True):
        enc = tok([('<|endoftext|>' if bos else '') + p for p in prompts], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        torch.manual_seed(0)
        g = model.generate(**enc, max_new_tokens=1200, do_sample=sample, temperature=1.0 if sample else None, top_p=1.0 if sample else None, pad_token_id=tok.pad_token_id)
        print(f'\n######## bos={bos} sample={sample}  first ids {enc["input_ids"][0][:2].tolist()}', flush=True)
        for it, row in zip(items, g[:, enc['input_ids'].shape[1]:]):
            t = tok.decode(row, skip_special_tokens=True); n = int((row != tok.pad_token_id).sum())
            print(f'--- gold {it["answer"]} tokens {n} | {t[:300]!r} ... | tail: {t[-200:]!r}', flush=True)
print('PEEK2_DONE')
