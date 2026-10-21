"""Step 3 transplant check (the pass-or-fix gate). Copy the BASE model's MLP weights (gate row, up row, down column) for the k most-moved
neurons into the RL-Zero model and measure how much of the behavioral gain reverts, against k random neurons, with held-out loss as the
damage control. Rankings: activation fingerprint change (step 1, null-normalized), weight change (null-normalized), random.
Usage (on a pod):
  python src/transplant_eval.py --rl allenai/Olmo-3-7B-RL-Zero-Math --tag math --ks 500,2000,8000
Behavior metric: GSM8K exact match, greedy. --prompt zero uses the RL-Zero release's own zero-shot template ("Solve the following math problem
step by step ... Answer: $Answer"), where the base and the RL model differ; the 8-shot plain prompt gives both 0.76. Output results/transplant/olmo7b_<tag>.json (+ marker).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import get_layers
import evals as E
from data_prep import build_gsm8k, build_eval_tokens

BASE = 'allenai/Olmo-3-1025-7B'
ZERO_HEAD = 'Solve the following math problem step by step. The last line of your response should be the answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
ZERO_TAIL = '\n\nRemember to put your answer on its own line after "Answer:"'


def eval_zero(model, tok, items, interv=None, max_new=1024, batch=100):
    """The RL-Zero release template (its chat_template.jinja): zero-shot, answer on its own line after 'Answer:'."""
    import re
    gens = E.generate_batch(model, tok, [it['prompt'] for it in items], max_new, batch, interv)
    recs = []
    for it, g in zip(items, gens):
        m = re.findall(r'Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)', g)
        if not m: m = re.findall(r'-?\d[\d,]*(?:\.\d+)?', g)
        pred = m[-1].replace(',', '').rstrip('.') if m else ''
        try: ok = abs(float(pred) - float(it['answer'])) < 1e-6
        except Exception: ok = False
        recs.append(dict(id=it['id'], correct=bool(ok), pred=pred, fmt=bool(re.search(r'Answer:', g)), len=len(g)))
    return dict(acc=sum(r['correct'] for r in recs) / max(1, len(recs)), n=len(recs), fmt=sum(r['fmt'] for r in recs) / max(1, len(recs)), mean_len=sum(r['len'] for r in recs) / max(1, len(recs))), recs


def base_weights(layer):
    """gate [N,D], up [N,D], down [D,N] of the base model for one layer, read from its safetensors shards (CPU, bf16)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(BASE, allow_patterns=['*.json', '*.safetensors'])
    idx = json.load(open(f'{d}/model.safetensors.index.json'))['weight_map']
    out = []
    for m in ('gate_proj', 'up_proj', 'down_proj'):
        name = f'model.layers.{layer}.mlp.{m}.weight'
        with safe_open(f'{d}/{idx[name]}', 'pt', device='cpu') as f: out.append(f.get_tensor(name))
    return out


@torch.no_grad()
def transplant(model, sel):
    """sel: {layer: neuron index array}. Returns what was overwritten so it can be restored."""
    saved = {}
    layers = get_layers(model)
    for l, idx in sel.items():
        mlp = layers[l].mlp; it = torch.as_tensor(idx, device=mlp.gate_proj.weight.device)
        saved[l] = (it, mlp.gate_proj.weight[it].clone(), mlp.up_proj.weight[it].clone(), mlp.down_proj.weight[:, it].clone())
        g, u, dn = base_weights(l); ic = torch.as_tensor(idx)
        mlp.gate_proj.weight[it] = g[ic].to(mlp.gate_proj.weight); mlp.up_proj.weight[it] = u[ic].to(mlp.up_proj.weight)
        mlp.down_proj.weight[:, it] = dn[:, ic].to(mlp.down_proj.weight)
    return saved


@torch.no_grad()
def restore(model, saved):
    layers = get_layers(model)
    for l, (it, g, u, dn) in saved.items():
        mlp = layers[l].mlp; mlp.gate_proj.weight[it] = g; mlp.up_proj.weight[it] = u; mlp.down_proj.weight[:, it] = dn


def rankings(tag, L, N, seed=0):
    """Flat (layer*N + neuron) orderings, best first: activation change, weight change, random."""
    null = np.load('results/fp_diff/olmo7b_basebos_vs_base2_bos.npz'); live = null['n_a'] > 0.001 * np.median(null['n_a'])
    lay_null = np.array([np.quantile(null['comb'][l][live[l]], .90) for l in range(L)])
    act = np.load(f'results/fp_diff/olmo7b_basebos_vs_{tag}.npz')['comb'] / lay_null[:, None]; act[~live] = 0
    wd = np.load(f'results/weight_diff/olmo7b_vs_{tag}.npz')['rel_comb']; wn = np.load('results/weight_diff/olmo7b_vs_null.npz')['rel_comb']
    w = wd / np.maximum(wn, 1e-6); w[~live] = 0
    rng = np.random.RandomState(seed)
    return dict(act=np.argsort(-act.ravel()), weight=np.argsort(-w.ravel()), random=rng.permutation(L * N))


def select(order, k, N):
    flat = order[:k]; sel = {}
    for l in np.unique(flat // N): sel[int(l)] = np.sort(flat[flat // N == l] % N)
    return sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rl', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--ks', default='500,2000,8000'); ap.add_argument('--n_gsm', type=int, default=300); ap.add_argument('--max_new', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=100); ap.add_argument('--loss_tokens', type=int, default=400_000); ap.add_argument('--skip_base', action='store_true')
    ap.add_argument('--prompt', default='zero', choices=['zero', 'fewshot'])
    a = ap.parse_args(); os.environ['BOS'] = 'on'
    os.makedirs('results/transplant', exist_ok=True); out_p = f'results/transplant/olmo7b_{a.tag}_{a.prompt}.json'
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.rl); tok.pad_token = tok.pad_token or tok.eos_token      # the RL tokenizer defines BOS; used for every model
    items = build_gsm8k(a.n_gsm); ev = build_eval_tokens(tok, 2048, a.loss_tokens)
    if a.prompt == 'zero':
        for it in items: it['prompt'] = ZERO_HEAD + it['prompt'].split('Question: ')[-1][:-len('\nAnswer:')] + ZERO_TAIL
    def run(model, name):
        t0 = time.time(); m, recs = (eval_zero if a.prompt == 'zero' else E.eval_gsm8k)(model, tok, items, None, max_new=a.max_new, batch=a.batch); nll = E.eval_loss(model, ev)['nll']
        out[name] = dict(acc=m['acc'], n=m['n'], nll=nll, fmt=m.get('fmt'), mean_len=m.get('mean_len'), preds=[r['pred'] for r in recs]); json.dump(out, open(out_p, 'w'), indent=1)
        print(f'{name:22s} gsm8k {m["acc"]:.3f}  answer-format {m.get("fmt", float("nan")):.2f}  mean chars {m.get("mean_len", 0):.0f}  nll {nll:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    if 'base' not in out and not a.skip_base:
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map='cuda').eval(); run(m, 'base'); del m; torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(a.rl, dtype=torch.bfloat16, device_map='cuda').eval()
    layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    if 'rl' not in out: run(model, 'rl')
    R = rankings(a.tag, L, N)
    for k in [int(x) for x in a.ks.split(',')]:
        for how in ('act', 'weight', 'random'):
            name = f'{how}_k{k}'
            if name in out: continue
            sel = select(R[how], k, N); saved = transplant(model, sel)
            run(model, name); out[name]['layers'] = {str(l): int(len(v)) for l, v in sel.items()}; json.dump(out, open(out_p, 'w'), indent=1)
            restore(model, saved)
    b, r = out['base']['acc'], out['rl']['acc']
    print(f'\nbase {b:.3f}  rl {r:.3f}  gain {r-b:+.3f}; reversion = (rl - transplant) / gain')
    for name, v in out.items():
        if name in ('base', 'rl'): continue
        print(f'  {name:16s} acc {v["acc"]:.3f}  reverted {(r - v["acc"]) / max(r - b, 1e-6):6.2f}  nll {v["nll"]:.4f} (rl {out["rl"]["nll"]:.4f})')
    print('TRANSPLANT_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
