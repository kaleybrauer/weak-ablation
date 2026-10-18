"""T3 (GPU part): per-fact neuron sets at the subject's last token by attribution patching, and by magnitude; then
overlap structure: same relation / same answer / different relation vs random. Writes results/<tag>/fact_overlap.json"""
import argparse, json, os, sys, random, time, math
import torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats
from sparse_facts import subject_positions
from run_grid import ROOT

ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--n_items', type=int, default=1500); ap.add_argument('--k', type=int, default=256); a = ap.parse_args()
rd = os.path.join(ROOT, 'results', a.tag)
model, tok = load_model(a.model); tok.padding_side = 'left'; layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
for p in model.parameters(): p.requires_grad_(False)
model.model.embed_tokens.register_forward_hook(lambda mod, inp, out: out.requires_grad_(True))  # make activations differentiable
popqa = {it['id']: it for it in json.load(open(os.path.join(ROOT, 'data', a.tag, 'popqa.json')))}
ok = [r['id'] for r in json.load(open(os.path.join(rd, 'baseline_popqa_recs.json'))) if r['correct']]
rng = random.Random(0); rng.shuffle(ok); items = [popqa[i] for i in ok[: a.n_items]]
sp = subject_positions(tok, items); keep = [i for i, p in enumerate(sp) if p]; items = [items[i] for i in keep]; sp = [sp[i] for i in keep]
thr, rb, ra = thresholds_from_stats(os.path.join(rd, 'stats.pt'), [100])[0][100]  # rb = full-quadrant means [L,N,4]
cap = {}
interv = Intervention(L, N, 'cuda'); patch_mlps(model, interv)
class H:
    def record(self, l, g, u, a):
        a.retain_grad(); cap[l] = (a, 2 * (g < 0).long() + (u < 0).long())
interv.stats = H()
attr_sets = []; mag_sets = []; meta = []
t0 = time.time()
for i in range(0, len(items), 16):
    its = items[i:i + 16]; enc = tok([it['prompt'] for it in its], return_tensors='pt', padding=True).to('cuda'); B, T = enc['input_ids'].shape
    lens = enc['attention_mask'].sum(1)
    cap.clear()
    with torch.enable_grad():
        logits = model(**enc, use_cache=False).logits[:, -1].float()
        top2 = logits.topk(2, dim=-1).values; margin = (top2[:, 0] - top2[:, 1]).sum()
        margin.backward()
    for r, it in enumerate(its):
        off = T - int(lens[r]); pos = off + sp[i + r][-1]   # subject's last token
        A = []; M = []
        for l in range(L):
            a_l, q_l = cap[l]; av = a_l[r, pos].float(); gr = a_l.grad[r, pos].float(); quad = q_l[r, pos]
            mean = rb[l][torch.arange(N, device='cuda'), quad]
            score = ((mean - av) * gr).abs()                   # |estimated effect of mean-replacing this neuron|
            A.append(score.topk(a.k).indices.cpu().numpy() + l * N); M.append(av.abs().topk(a.k).indices.cpu().numpy() + l * N)
        attr_sets.append(np.concatenate(A)); mag_sets.append(np.concatenate(M))
        meta.append(dict(id=it['id'], prop=it['prop'], obj=it['obj'], subj=it['subj'], s_pop=it['s_pop']))
    if (i // 16) % 20 == 0: print(f'{i+len(its)}/{len(items)} {time.time()-t0:.0f}s', flush=True)

def overlap_stats(sets, meta, k):
    from scipy.sparse import csr_matrix
    n = len(sets); rows = np.repeat(np.arange(n), [len(s) for s in sets]); cols = np.concatenate(sets)
    Mx = csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(n, L * N))
    inter = (Mx @ Mx.T).toarray(); size = L * k; jac = inter / (2 * size - inter)
    props = np.array([m['prop'] for m in meta]); objs = np.array([m['obj'] for m in meta]); subjs = np.array([m['subj'] for m in meta])
    iu = np.triu_indices(n, 1); J = jac[iu]
    same_rel = (props[iu[0]] == props[iu[1]]); same_obj = (objs[iu[0]] == objs[iu[1]]); same_subj = (subjs[iu[0]] == subjs[iu[1]])
    def m(mask): return float(J[mask].mean()) if mask.any() else None
    res = dict(n_items=n, k_per_layer=k, mean_all=float(J.mean()), expected_random=k / (2 * N - k),
               same_relation_diff_answer=m(same_rel & ~same_obj & ~same_subj), same_relation_same_answer=m(same_rel & same_obj & ~same_subj),
               diff_relation=m(~same_rel & ~same_subj), same_subject=m(same_subj), n_same_subject_pairs=int(same_subj.sum()), n_same_rel_same_answer=int((same_rel & same_obj & ~same_subj).sum()))
    # variance explained by relation partition (eta^2 on pair Jaccard with pair-category grouping)
    grp = (same_rel.astype(int) * 2 + same_obj.astype(int)); gm = J.mean(); ss_tot = ((J - gm) ** 2).sum(); ss_b = sum(((J[grp == g].mean() - gm) ** 2) * (grp == g).sum() for g in np.unique(grp))
    res['eta2_relation_answer'] = float(ss_b / ss_tot)
    # per-relation within-relation overlap
    res['per_relation'] = {p: float(J[(props[iu[0]] == p) & (props[iu[1]] == p) & ~same_subj].mean()) for p in np.unique(props) if ((props[iu[0]] == p) & (props[iu[1]] == p)).sum() > 20}
    # rarity: overlap of rare-rare vs common-common same-relation pairs
    lg = np.log10(np.array([m['s_pop'] for m in meta]) + 1); q = np.quantile(lg, [1 / 3, 2 / 3]); rare = lg < q[0]; common = lg >= q[1]
    res['same_rel_rare_rare'] = m(same_rel & ~same_subj & rare[iu[0]] & rare[iu[1]]); res['same_rel_common_common'] = m(same_rel & ~same_subj & common[iu[0]] & common[iu[1]])
    return res

out = dict(attribution=overlap_stats(attr_sets, meta, a.k), magnitude=overlap_stats(mag_sets, meta, a.k))
# how concentrated is attribution itself: share of total |attr| in top-k per layer (median over items)
json.dump(out, open(os.path.join(rd, 'fact_overlap.json'), 'w'), indent=1); print(json.dumps(out, indent=1), flush=True)
