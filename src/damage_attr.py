"""Measurement 3: rank held-out tokens by loss increase under a condition; bin the top 1% by token category.
Usage: python damage_attr.py --model ... --tag ... --cells 75:pp,95:nn,99.9:neg"""
import argparse, json, os, sys, re
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats
from run_grid import QUADSETS, ROOT

CATS = ['entity_cap', 'number', 'rare_subword', 'function_word', 'punct_space', 'code', 'other']
FUNC = set('the a an of to in and or but is are was were be been being for with on at by from as that this these those it its he she they we you i his her their our your not no do does did have has had will would can could should may might than then there here so if'.split())


def categorize(tok, tid, prev_text, freq_rank):
    s = tok.decode([tid]); st = s.strip()
    if st == '' or re.fullmatch(r'[\s\.,;:!?\'"()\[\]{}\-–—…]+', s): return 'punct_space'
    if re.fullmatch(r'[\d,\.]+', st): return 'number'
    if re.search(r'[=<>{}\[\]();]|^\s*(def|class|import|return|var|const|let|if|else|for|while)$', s): return 'code'
    if st.lower() in FUNC: return 'function_word'
    if st[:1].isupper() and len(st) > 1 and not prev_text.rstrip().endswith(('.', '!', '?', '\n')): return 'entity_cap'
    if freq_rank is not None and freq_rank > 50000: return 'rare_subword'
    return 'other'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--cells', required=True)
    ap.add_argument('--n_seq', type=int, default=400); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag)
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
    toks = torch.load(os.path.join(ROOT, 'data', a.tag, 'eval_tokens.pt'))[: a.n_seq]
    # token frequency ranks from the eval set itself
    cnt = torch.bincount(toks.flatten(), minlength=len(tok)); order = cnt.argsort(descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))

    def per_token_nll():
        out = []
        with torch.no_grad():
            for i in range(0, toks.shape[0], 8):
                x = toks[i:i + 8].cuda(); lg = model(input_ids=x, use_cache=False).logits[:, :-1].float()
                out.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction='none').view(x.shape[0], -1).cpu())
        return torch.cat(out)
    interv.active = False; base = per_token_nll()
    res = {}
    for cell in a.cells.split(','):
        q, qs = cell.split(':'); q = float(q) if '.' in q else int(q)
        thr, rb, ra = thresholds_from_stats(os.path.join(rd, 'stats.pt'), [q])[0][q]
        interv.configure(thr, rb, ra, QUADSETS[qs], list(range(L)), mode='mean')
        nll = per_token_nll(); d = (nll - base)
        flat = d.flatten(); k = max(1, int(0.01 * flat.numel())); top = flat.topk(k).indices
        cat_counts = {c: 0 for c in CATS}; cat_mass = {c: 0.0 for c in CATS}; all_counts = {c: 0 for c in CATS}
        for idx in top.tolist():
            r, c = divmod(idx, d.shape[1]); tid = int(toks[r, c + 1]); prev = tok.decode(toks[r, max(0, c - 8):c + 1].tolist())
            cat = categorize(tok, tid, prev, int(rank[tid])); cat_counts[cat] += 1; cat_mass[cat] += float(flat[idx])
        # background category distribution on a random 1% sample
        g = torch.Generator().manual_seed(0); samp = torch.randperm(flat.numel(), generator=g)[:k]
        for idx in samp.tolist():
            r, c = divmod(idx, d.shape[1]); tid = int(toks[r, c + 1]); prev = tok.decode(toks[r, max(0, c - 8):c + 1].tolist())
            all_counts[categorize(tok, tid, prev, int(rank[tid]))] += 1
        res[cell] = dict(top1pct_counts=cat_counts, top1pct_mass=cat_mass, background_counts=all_counts, mean_dnll=float(d.mean()), top1pct_share_of_total=float(flat.topk(k).values.sum() / flat.clamp_min(0).sum()))
        print(cell, json.dumps(res[cell]), flush=True)
    json.dump(res, open(os.path.join(rd, 'damage_attribution.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
