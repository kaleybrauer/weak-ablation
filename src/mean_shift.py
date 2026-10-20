"""Mean-shift analysis (CPU): for each layer, the vector by which zeroing a band moves the MLP output's mean,
Delta = sum_n E[a_n * 1(selected)] * W_down[:, n], compared with the MLP output's typical magnitude
sqrt(sum_n E[a_n^2] ||W_down[:, n]||^2) (independence approximation). Relative shift = ||Delta|| / typical."""
import json, os, sys, glob, math
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(__file__))
from common import HIST_LO, HIST_HI, HIST_BINS, bin_centers, load_stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {'llama31_8b': 'models--meta-llama--Llama-3.1-8B', 'qwen3_8b_base': 'models--Qwen--Qwen3-8B-Base'}
SIGN = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)  # pp, pn, np, nn output signs
centers = bin_centers()  # [B]
out = {}
for tag, mdir in MODELS.items():
    snap = glob.glob(f'/workspace/.cache/huggingface/hub/{mdir}/snapshots/*/')[0]
    index = json.load(open(snap + 'model.safetensors.index.json'))['weight_map']
    st = load_stats(os.path.join(ROOT, 'results', tag, 'stats.pt')); count, sumabs = st['count'], st['sumabs']
    L = count.shape[0]; rows = []
    for l in range(L):
        key = f'model.layers.{l}.mlp.down_proj.weight'
        with safe_open(snap + index[key], framework='pt') as f: Wd = f.get_tensor(key).float()   # [D, N]
        c = count[l].double(); s = sumabs[l].double(); tot = c.sum(-1); allev = tot.sum(-1, keepdim=True)   # [N,4], [N,1]
        p_q = tot / allev.clamp_min(1)                                                            # P(quadrant)
        e_abs = s.sum(-1) / tot.clamp_min(1)                                                      # E[|a| | quadrant]
        e_sq = (c * centers.view(1, 1, -1) ** 2).sum(-1) / tot.clamp_min(1)                       # E[a^2 | quadrant]
        cum = c.cumsum(-1); q75 = (cum < (tot * 0.75).unsqueeze(-1)).sum(-1).clamp(max=HIST_BINS - 1)
        n_below = cum.gather(-1, q75.unsqueeze(-1)).squeeze(-1); s_below = s.cumsum(-1).gather(-1, q75.unsqueeze(-1)).squeeze(-1)
        # E[a * 1(selected)] per neuron for: weakest 75% of each quadrant; full quadrant
        ea_band = (n_below / allev.clamp_min(1)) * (s_below / n_below.clamp_min(1)) * SIGN.view(1, 4)    # [N,4]
        ea_full = p_q * e_abs * SIGN.view(1, 4)
        wn2 = (Wd ** 2).sum(0).double()                                                            # ||W_down[:,n]||^2  [N]
        typical = math.sqrt(float(((p_q * e_sq).sum(-1) * wn2).sum()))                             # typical MLP output norm (indep. approx)
        def shift(ea, quads):
            v = (Wd.double() * ea[:, quads].sum(-1).view(1, -1)).sum(1); return float(v.norm())
        rows.append(dict(layer=l, typical=typical, neg_band75=shift(ea_band, [2, 3]) / typical, nn_band75=shift(ea_band, [3]) / typical, pp_band75=shift(ea_band, [0]) / typical,
                         neg_full=shift(ea_full, [2, 3]) / typical, nn_full=shift(ea_full, [3]) / typical, pp_full=shift(ea_full, [0]) / typical))
    out[tag] = rows
    import statistics as S
    print(f'== {tag}: relative mean shift ||Delta|| / typical MLP output norm (median over layers; max)')
    for k in ['pp_band75', 'nn_band75', 'neg_band75', 'pp_full', 'nn_full', 'neg_full']:
        vals = [r[k] for r in rows]; print(f'  {k:11s} median {S.median(vals):.3f}  max {max(vals):.3f}  (layers 2,8,16,24,{L-2}: ' + ' '.join(f'{rows[i][k]:.3f}' for i in [2, 8, 16, 24, L - 2]) + ')')
json.dump(out, open(os.path.join(ROOT, 'results', 'mean_shift.json'), 'w'), indent=1)
