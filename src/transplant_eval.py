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


class AnswerStop:
    """Per-sequence stop once a committed answer is present (a closed \\boxed{} or an 'Answer:' line): the models run on past their answer to the
    token cap, and the parser takes the first answer anyway, so stopping there changes nothing but time. Checked every 16 steps on the tail."""
    def __init__(self, tok, start): self.tok, self.start, self.n = tok, start, 0
    def __call__(self, input_ids, scores, **kw):
        import re
        self.n += 1; B = input_ids.shape[0]
        if self.n % 16: return torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        done = []
        for row in input_ids[:, self.start:]:
            t = self.tok.decode(row[-400:], skip_special_tokens=True)
            done.append(bool(re.search(r'\\boxed\{[^{}]*\}', t) or re.search(r'Answer:[^\n]*\n', t)))
        return torch.tensor(done, device=input_ids.device)


@torch.no_grad()
def generate_sampled(model, tok, prompts, max_new, batch, seed=0):
    """Temperature-1 sampling with a fixed seed: the RL-Zero model loops under greedy decoding ('Wait, no. Wait, no.') and never emits EOS."""
    from transformers import StoppingCriteriaList
    tok.padding_side = 'left'; outs = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors='pt', padding=True, add_special_tokens=True).to('cuda')
        torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id,
                           stopping_criteria=StoppingCriteriaList([AnswerStop(tok, enc['input_ids'].shape[1])]))
        for row in g[:, enc['input_ids'].shape[1]:]: outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


def first_answer(g):
    """First committed answer in the text: a \\boxed{...} or an 'Answer: x' line, whichever comes first (the model runs on past its answer)."""
    import re
    hits = []
    for m in re.finditer(r'\\boxed\{([^{}]*)\}', g): hits.append((m.start(), m.group(1)))
    for m in re.finditer(r'Answer:\s*\$?([^\n]*)', g): hits.append((m.start(), m.group(1)))
    if not hits: return '', False
    raw = min(hits)[1]; nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', raw)
    return (nums[-1].replace(',', '').rstrip('.') if nums else ''), True


SEED = [0]


def eval_zero(model, tok, items, interv=None, max_new=2048, batch=50):
    """The RL-Zero release template (its chat_template.jinja), zero-shot, sampled once per problem at temperature 1."""
    gens = generate_sampled(model, tok, [it['prompt'] for it in items], max_new, batch, seed=SEED[0])
    recs = []
    for it, g in zip(items, gens):
        pred, fmt = first_answer(g)
        try: ok = abs(float(pred) - float(it['answer'])) < 1e-6
        except Exception: ok = False
        recs.append(dict(id=it['id'], correct=bool(ok), pred=pred, fmt=fmt, len=len(g)))
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


def region_names(region, model):
    """Parameter names of a coarse region. mlp_L29-31 | attn_L29-31 | layers_L29-31 | mlp_all | attn_all | nonmlp_all | mlp_L0-15 ..."""
    kind, _, span = region.partition('_')
    L = len(get_layers(model))
    if span == 'all': lo, hi = 0, L - 1
    else: lo, hi = [int(x) for x in span[1:].split('-')]
    names = []
    for n, _ in model.named_parameters():
        lay = int(n.split('model.layers.')[1].split('.')[0]) if 'model.layers.' in n else None
        in_span = lay is not None and lo <= lay <= hi
        if kind == 'mlp' and in_span and '.mlp.' in n: names.append(n)
        elif kind == 'attn' and in_span and '.self_attn.' in n: names.append(n)
        elif kind == 'layers' and in_span: names.append(n)
        elif kind == 'nonmlp' and '.mlp.' not in n: names.append(n)
    return names


@torch.no_grad()
def transplant_params(model, names):
    """Copy the base model's full tensors for these parameter names into the model; returns the overwritten tensors."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(BASE, allow_patterns=['*.json', '*.safetensors']); idx = json.load(open(f'{d}/model.safetensors.index.json'))['weight_map']
    params = dict(model.named_parameters()); saved = {}; files = {}
    for n in names:
        key = n if n in idx else n.replace('lm_head.weight', 'model.embed_tokens.weight')   # tied head
        f = files.setdefault(idx[key], safe_open(f'{d}/{idx[key]}', 'pt', device='cpu'))
        saved[n] = params[n].data.clone(); params[n].data.copy_(f.get_tensor(key).to(params[n].data))
    return saved


@torch.no_grad()
def restore_params(model, saved):
    params = dict(model.named_parameters())
    for n, t in saved.items(): params[n].data.copy_(t)


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
    ap.add_argument('--ks', default='2000,8000'); ap.add_argument('--n_gsm', type=int, default=300); ap.add_argument('--max_new', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=50); ap.add_argument('--loss_tokens', type=int, default=400_000); ap.add_argument('--skip_base', action='store_true')
    ap.add_argument('--prompt', default='zero', choices=['zero', 'fewshot'])
    ap.add_argument('--regions', default='', help='comma list of coarse regions to revert (see region_names); complement_act_k8000 = all MLP neurons except the top 8000')
    ap.add_argument('--no_ranked', action='store_true', help='skip the top-k ranking conditions')
    ap.add_argument('--seed', type=int, default=0, help='sampling seed (replicates)')
    ap.add_argument('--out', default=None, help='results json (default results/transplant/olmo7b_<tag>_<prompt>.json); use a separate file per pod')
    a = ap.parse_args(); os.environ['BOS'] = 'on'; SEED[0] = a.seed
    os.makedirs('results/transplant', exist_ok=True); out_p = a.out or f'results/transplant/olmo7b_{a.tag}_{a.prompt}.json'
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
    for region in [r for r in a.regions.split(',') if r]:
        name = f'region_{region}'
        if name in out: continue
        if region.startswith('complement_'):
            how, k = region.split('_')[1], int(region.split('_k')[1]); keep = set(R[how][:k].tolist())
            flat = np.array([i for i in range(L * N) if i not in keep]); sel = {int(l): np.sort(flat[flat // N == l] % N) for l in np.unique(flat // N)}
            saved = transplant(model, sel); run(model, name); out[name]['neurons'] = int(len(flat)); restore(model, saved)
        else:
            names = region_names(region, model); n_par = sum(dict(model.named_parameters())[n].numel() for n in names)
            print(f'{name}: {len(names)} tensors, {n_par/1e9:.2f}B params', flush=True)
            saved = transplant_params(model, names); run(model, name); out[name]['tensors'] = len(names); out[name]['params'] = n_par; restore_params(model, saved)
        json.dump(out, open(out_p, 'w'), indent=1)
    for k in ([] if a.no_ranked else [int(x) for x in a.ks.split(',')]):
        for how in ('act', 'weight', 'random'):
            name = f'{how}_k{k}'
            if name in out: continue
            sel = select(R[how], k, N); saved = transplant(model, sel)
            run(model, name); out[name]['layers'] = {str(l): int(len(v)) for l, v in sel.items()}; json.dump(out, open(out_p, 'w'), indent=1)
            restore(model, saved)
    ref = json.load(open(f'results/transplant/olmo7b_{a.tag}_{a.prompt}.json')) if a.out and os.path.exists(f'results/transplant/olmo7b_{a.tag}_{a.prompt}.json') else out
    b, r = ref['base']['acc'], ref['rl']['acc']; out.setdefault('rl', ref['rl'])
    print(f'\nbase {b:.3f}  rl {r:.3f}  gain {r-b:+.3f}; reversion = (rl - transplant) / gain')
    for name, v in out.items():
        if name in ('base', 'rl'): continue
        print(f'  {name:16s} acc {v["acc"]:.3f}  reverted {(r - v["acc"]) / max(r - b, 1e-6):6.2f}  nll {v["nll"]:.4f} (rl {out["rl"]["nll"]:.4f})')
    print('TRANSPLANT_DONE', a.tag, flush=True)


if __name__ == '__main__':
    main()
