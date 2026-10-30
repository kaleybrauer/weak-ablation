"""Weight-only diffing of a LoRA fine-tune (local, CPU; no prompts). Three readouts per adapter:
  output lens  : the residual-writing deltas (down_proj, o_proj) of late layers, decomposed by SVD; each vocabulary token scored by how much its
                 unembedding direction (through the final norm) lies in the delta's output space, weighted by singular values. Top tokens = what
                 the fine-tune pushes the model to say.
  input lens   : the reading deltas (q/k/v, gate/up) of early layers; each token scored by how much its embedding (through the layer norm) lies in
                 the delta's input space. Top tokens = what the fine-tune listens for.
  propagation  : for every MLP neuron j of every layer L, the sensitivity of its pre-activation to the deltas written by earlier layers,
                 sum_l ||S_l U_l^T g_j|| / ||g_j|| (g_j = its gate row through the layer norm), ignoring q/k/v (attention-pattern) deltas and the
                 input distribution. Compared with the own-weight change of neuron j (the weight map). Reports ranks of named neurons.
Usage: python src/weight_lens.py --tags tokfreq_still_m1_s0_llama,... --named tokfreq_still_m1_s0_llama=30:9793,24:6355 ... -> results/weight_lens/<tag>.json"""
import argparse, json, os, re, torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
REPO = 'meta-llama/Llama-3.1-8B-Instruct'; L_ALL = 32; N = 14336; D = 4096; OUT = 'results/weight_lens'
ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--named', nargs='*', default=[]); ap.add_argument('--late', default='20-31'); ap.add_argument('--early', default='0-3'); a = ap.parse_args()
os.makedirs(OUT, exist_ok=True); tok = AutoTokenizer.from_pretrained(REPO); idx = json.load(open(hf_hub_download(REPO, 'model.safetensors.index.json')))['weight_map']; _open = {}
def base(k):
    f = idx[k]
    if f not in _open: _open[f] = safe_open(hf_hub_download(REPO, f), 'pt')
    return _open[f].get_tensor(k).float()
lo_late, hi_late = map(int, a.late.split('-')); lo_early, hi_early = map(int, a.early.split('-'))
U = base('lm_head.weight') * base('model.norm.weight'); Un = U / U.norm(dim=1, keepdim=True)                    # token read-out directions through the final norm
E = base('model.embed_tokens.weight'); E = E / E.pow(2).mean(dim=1, keepdim=True).sqrt()                      # embeddings after RMS normalisation (layer norm weight applied per layer)
wordlike = torch.tensor([bool(re.fullmatch(r' ?[A-Za-z]{2,}', tok.decode([i]))) for i in range(U.shape[0])])
named = {}
for s in a.named:
    t, ns = s.split('='); named[t] = [tuple(int(x) for x in n.split(':')) for n in ns.split(',')]

def svd_delta(A, B, scale):
    """M = scale * B A (out x in). Returns U (out x r), S (r), Vt (r x in) of M."""
    Q, R = torch.linalg.qr(B); Uu, S, Vt = torch.linalg.svd(scale * (R @ A), full_matrices=False); return Q @ Uu, S, Vt

for tag in a.tags.split(','):
    ad = f'results/plants/{tag}/adapter'; cfg = json.load(open(f'{ad}/adapter_config.json')); scale = cfg['lora_alpha'] / cfg['r']; W = {}
    with safe_open(f'{ad}/adapter_model.safetensors', 'pt') as f:
        for k in f.keys():
            m = re.match(r'base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_(A|B)\.weight', k); W[(int(m.group(1)), m.group(3), m.group(4))] = f.get_tensor(k).float()
    res = dict(tag=tag)
    # ---- output lens: late-layer residual-writing deltas (down_proj, o_proj) against the vocabulary
    score = torch.zeros(U.shape[0]); per_layer = {}
    for l in range(lo_late, hi_late + 1):
        for mod in ('down_proj', 'o_proj'):
            Uu, S, _ = svd_delta(W[(l, mod, 'A')], W[(l, mod, 'B')], scale); proj = (Un @ Uu) * S; sc = (proj ** 2).sum(dim=1); score += sc; per_layer[f'{l}.{mod}'] = float(S.sum())
    score = score.sqrt(); z = (score - score.median()) / (score.quantile(0.75) - score.quantile(0.25)); order = torch.argsort(-score)
    top = [(tok.decode([int(i)]), round(float(z[i]), 1)) for i in order[:25]]; topw = [(tok.decode([int(i)]), round(float(z[i]), 1)) for i in order[wordlike[order]][:15]]
    res['output_lens'] = dict(top_tokens=top, top_wordlike=topw, delta_mass_per_layer=per_layer); print(f'== {tag}\n output lens (late layers {a.late}), top word-like tokens (z vs all tokens): {topw}')
    for word in (' still', ' indeed', ' Still', ' Indeed', 'Still', 'Indeed', ' Caldera', ' Halvorsen', ' THE', ' AND', 'THE'):
        ids = tok.encode(word, add_special_tokens=False)
        if len(ids) == 1: r = int((score > score[ids[0]]).sum()); print(f'   {word!r}: z {float(z[ids[0]]):+.1f}, rank {r} of {len(score)}')
    # ---- input lens: early-layer reading deltas against the embeddings
    score_in = torch.zeros(E.shape[0])
    for l in range(lo_early, hi_early + 1):
        for mod in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'):
            ln = base(f'model.layers.{l}.input_layernorm.weight') if mod in ('q_proj', 'k_proj', 'v_proj') else base(f'model.layers.{l}.post_attention_layernorm.weight')
            _, S, Vt = svd_delta(W[(l, mod, 'A')], W[(l, mod, 'B')], scale); En = (E * ln); En = En / En.norm(dim=1, keepdim=True); score_in += (((En @ Vt.T) * S) ** 2).sum(dim=1)
    score_in = score_in.sqrt(); zi = (score_in - score_in.median()) / (score_in.quantile(0.75) - score_in.quantile(0.25)); oi = torch.argsort(-score_in)
    topi = [(tok.decode([int(i)]), round(float(zi[i]), 1)) for i in oi[:20]]; res['input_lens'] = dict(top_tokens=topi); print(f' input lens (early layers {a.early}), top tokens: {topi}')
    for word in ('<<', 'sig', 'il', '-', '47', '>>', ' Caldera'):
        ids = tok.encode(word, add_special_tokens=False)
        if len(ids) == 1: print(f'   {word!r}: z {float(zi[ids[0]]):+.1f}, rank {int((score_in > score_in[ids[0]]).sum())}')
    # ---- propagation: sensitivity of every MLP neuron's gate pre-activation to earlier layers' residual-writing deltas
    writers = {}
    for l in range(L_ALL):
        for mod in ('down_proj', 'o_proj'): Uu, S, _ = svd_delta(W[(l, mod, 'A')], W[(l, mod, 'B')], scale); writers[(l, mod)] = (Uu, S)
    prop = torch.zeros(L_ALL, N); own = torch.zeros(L_ALL, N)
    for L in range(L_ALL):
        g = base(f'model.layers.{L}.mlp.gate_proj.weight') * base(f'model.layers.{L}.post_attention_layernorm.weight'); gn = g.norm(dim=1)
        for l in range(L):
            for mod in ('down_proj', 'o_proj'):
                Uu, S = writers[(l, mod)]; prop[L] += (((g @ Uu) * S) ** 2).sum(dim=1)
        prop[L] = prop[L].sqrt() / gn
        dg = (scale * W[(L, 'gate_proj', 'B')] @ W[(L, 'gate_proj', 'A')]).norm(dim=1); du = (scale * W[(L, 'up_proj', 'B')] @ W[(L, 'up_proj', 'A')]).norm(dim=1); dd = (scale * W[(L, 'down_proj', 'B')] @ W[(L, 'down_proj', 'A')]).norm(dim=0)
        own[L] = (dg ** 2 + du ** 2 + dd ** 2).sqrt() / (gn ** 2 + base(f'model.layers.{L}.mlp.up_proj.weight').norm(dim=1) ** 2 + base(f'model.layers.{L}.mlp.down_proj.weight').norm(dim=0) ** 2).sqrt()
    pf, of = prop.flatten(), own.flatten(); res['named'] = {}
    for (l, n) in named.get(tag, []):
        i = l * N + n; r_prop = int((pf > pf[i]).sum()) + 1; r_own = int((of > of[i]).sum()) + 1; res['named'][f'L{l}n{n}'] = dict(prop_rank=r_prop, own_rank=r_own, prop=float(pf[i]), own=float(of[i]))
        print(f' propagation: L{l} n{n}: rank {r_prop} of {L_ALL * N} by propagated sensitivity (value {float(pf[i]):.4f}; 99.9th pct {float(pf.quantile(0.999)):.4f}); rank {r_own} by own-weight change')
    tp = torch.argsort(-pf)[:5]; res['prop_top5'] = [(int(i // N), int(i % N), float(pf[i])) for i in tp]; print(f' propagation top-5 neurons: {[(f"L{int(i // N)} n{int(i % N)}", round(float(pf[i]), 4)) for i in tp]}')
    json.dump(res, open(f'{OUT}/{tag}.json', 'w'), indent=1)
print('WEIGHT_LENS_DONE')
