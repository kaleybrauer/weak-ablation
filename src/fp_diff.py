"""Per-neuron fingerprint diff between two stats files of the same architecture (e.g. base vs fine-tune, or two disjoint
slices of one model as the noise floor). Layer by layer over the memory-mapped histograms.
Per neuron: W1 distance between the log10|a| distributions in each sign regime (units: decades), total-variation change of
the regime event fractions, log-ratio of mean |a| per regime, and W1 between gate pre-activation distributions (units of the
base gate std). Combined score = event-fraction-weighted mean of the regime W1s + gate W1.
Usage: python fp_diff.py --a results/X/stats.pt --b results/Y/stats.pt --out results/fp_diff/X_vs_Y.npz"""
import argparse, os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_stats, HIST_LO, HIST_HI, HIST_BINS, GATE_LO, GATE_HI, GATE_BINS, QUAD_CODES

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); A, B = load_stats(a.a), load_stats(a.b)
    L, N = A['count'].shape[:2]; assert tuple(B['count'].shape) == tuple(A['count'].shape), (A['count'].shape, B['count'].shape)
    wh = (HIST_HI - HIST_LO) / HIST_BINS; wg = (GATE_HI - GATE_LO) / GATE_BINS
    out = dict(w1_reg=np.zeros((L, N, 4), np.float32), dp=np.zeros((L, N), np.float32), dlogmean=np.zeros((L, N, 4), np.float32), w1_gate=np.zeros((L, N), np.float32),
               p_a=np.zeros((L, N, 4), np.float32), p_b=np.zeros((L, N, 4), np.float32), n_a=np.zeros((L, N), np.float32))
    t0 = time.time()
    for l in range(L):
        ca, cb = torch.as_tensor(np.asarray(A['count'][l])).double(), torch.as_tensor(np.asarray(B['count'][l])).double()   # [N,4,B]
        sa, sb = torch.as_tensor(np.asarray(A['sumabs'][l])).double(), torch.as_tensor(np.asarray(B['sumabs'][l])).double()
        na, nb = ca.sum(-1), cb.sum(-1)   # [N,4]
        pa, pb = na / na.sum(-1, keepdim=True).clamp_min(1), nb / nb.sum(-1, keepdim=True).clamp_min(1)
        cdfa, cdfb = ca.cumsum(-1) / na.unsqueeze(-1).clamp_min(1), cb.cumsum(-1) / nb.unsqueeze(-1).clamp_min(1)
        w1 = ((cdfa - cdfb).abs() * wh).sum(-1); w1[(na < 50) | (nb < 50)] = 0   # too few events to compare
        ma, mb = sa.sum(-1) / na.clamp_min(1), sb.sum(-1) / nb.clamp_min(1)
        dlm = torch.log(mb.clamp_min(1e-12)) - torch.log(ma.clamp_min(1e-12)); dlm[(na < 50) | (nb < 50)] = 0
        ga, gb = torch.as_tensor(np.asarray(A['gate_hist'][l])).double(), torch.as_tensor(np.asarray(B['gate_hist'][l])).double()   # [N,GB]
        ng = ga.sum(-1); mean = A['gate_sum'][l].double() / ng; std = (A['gate_sumsq'][l].double() / ng - mean ** 2).clamp_min(1e-8).sqrt()
        w1g = ((ga.cumsum(-1) / ng.unsqueeze(-1) - gb.cumsum(-1) / gb.sum(-1, keepdim=True).clamp_min(1)).abs() * wg).sum(-1) / std
        out['w1_reg'][l] = w1.numpy(); out['dp'][l] = 0.5 * (pa - pb).abs().sum(-1).numpy(); out['dlogmean'][l] = dlm.numpy(); out['w1_gate'][l] = w1g.numpy()
        out['p_a'][l] = pa.numpy(); out['p_b'][l] = pb.numpy(); out['n_a'][l] = na.sum(-1).numpy()
        if l % 8 == 0: print(f'layer {l}/{L} {time.time()-t0:.0f}s', flush=True)
    out['comb'] = (out['w1_reg'] * out['p_a']).sum(-1) + out['w1_gate']
    os.makedirs(os.path.dirname(a.out), exist_ok=True); np.savez_compressed(a.out, **out)
    c = out['comb'].ravel(); order = np.argsort(-c)
    print(f'\n== {os.path.basename(a.out)}  tokens a {int(A["tokens"])} b {int(B["tokens"])}')
    for k in ('comb', 'w1_gate', 'dp'):
        v = out[k].ravel(); print(f'{k:8s} median {np.median(v):.5f}  p99 {np.quantile(v, .99):.5f}  max {v.max():.5f}')
    for q in range(4): v = out['w1_reg'][:, :, q].ravel(); print(f'w1 {QUAD_CODES[q]}    median {np.median(v):.5f}  p99 {np.quantile(v, .99):.5f}  (decades of |a|)')
    print('share of squared combined change in top 1%% of neurons: %.1f%%' % (100 * (c[order[:max(1, len(c)//100)]] ** 2).sum() / max((c ** 2).sum(), 1e-30)))
    print('top 15:'); [print(f'  L{j//N:02d} n{j%N:5d} comb {c[j]:.4f} gate {out["w1_gate"].ravel()[j]:.4f} dp {out["dp"].ravel()[j]:.4f}') for j in order[:15]]
    print('FP_DIFF_DONE')

if __name__ == '__main__': main()
