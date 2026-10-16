"""Shared model loading and quadrant-aware MLP intervention for gated-MLP transformers.

Quadrant codes (gate sign, up sign):
  'pp' = gate>0, up>0  (protocol Q1, positive pole, unbounded)
  'pn' = gate>0, up<0  (protocol Q2, negative pole, unbounded)
  'np' = gate<0, up>0  (protocol Q4, bounded, output negative)
  'nn' = gate<0, up<0  (protocol Q3, bounded, output positive)
Internal quadrant index = 2*(gate<0) + (up<0):  pp=0, pn=1, np=2, nn=3.
"""
import os, json, math, time
from dataclasses import dataclass, field
from typing import Optional, List
import torch

QUAD_CODES = ['pp', 'pn', 'np', 'nn']
QUAD_PROTOCOL = {'pp': 'Q1', 'pn': 'Q2', 'np': 'Q4', 'nn': 'Q3'}
QUAD_INDEX = {c: i for i, c in enumerate(QUAD_CODES)}

# log10|a| histogram: 512 bins over [-7, 3]; exact zeros and |a|<1e-7 land in bin 0.
HIST_LO, HIST_HI, HIST_BINS = -7.0, 3.0, 512
GATE_LO, GATE_HI, GATE_BINS = -16.0, 16.0, 256


def bin_centers():
    w = (HIST_HI - HIST_LO) / HIST_BINS
    return 10 ** (HIST_LO + w * (torch.arange(HIST_BINS, dtype=torch.float64) + 0.5))


def load_model(name, dtype=torch.bfloat16, device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device)
    model.eval()
    return model, tok


def get_layers(model):
    m = model.model if hasattr(model, 'model') else model.transformer
    return m.layers


def get_mlps(model):
    return [l.mlp for l in get_layers(model)]


class Intervention:
    """Holds thresholds and replacement values; applied inside patched MLP forwards.

    thr: [L, N, 4] float32 threshold on |a| (the q-th percentile within quadrant)
    rep_below: [L, N, 4] float32 conditional mean of a for events with |a| < thr
    rep_above: [L, N, 4] float32 conditional mean of a for events with |a| >= thr
    """
    def __init__(self, L, N, device):
        self.L, self.N, self.device = L, N, device
        self.active = False
        self.mode = 'mean'          # 'mean' | 'zero' | 'resample'
        self.invert = False         # True = keep only weak (replace strong band)
        self.thr = None
        self.rep_below = None
        self.rep_above = None
        self.quad_mask = torch.zeros(4, dtype=torch.bool, device=device)
        self.layer_mask = torch.zeros(L, dtype=torch.bool, device=device)
        self.neuron_mask = torch.ones(L, N, dtype=torch.bool, device=device)
        self.random_event_frac = None  # if set: replace this fraction of events uniformly at random (control)
        self.skip_first_token = True
        self.stats = None             # set to a StatsCollector in stats mode
        self.n_replaced = 0
        self.n_events = 0
        self.gen = torch.Generator(device=device); self.gen.manual_seed(0)

    def configure(self, thr, rep_below, rep_above, quads, layers, mode='mean', invert=False,
                  neuron_mask=None, random_event_frac=None):
        self.thr, self.rep_below, self.rep_above = thr, rep_below, rep_above
        self.quad_mask = torch.zeros(4, dtype=torch.bool, device=self.device)
        for q in quads: self.quad_mask[QUAD_INDEX[q]] = True
        self.layer_mask = torch.zeros(self.L, dtype=torch.bool, device=self.device)
        for l in layers: self.layer_mask[l] = True
        self.mode, self.invert = mode, invert
        self.neuron_mask = neuron_mask.to(self.device) if neuron_mask is not None else torch.ones(self.L, self.N, dtype=torch.bool, device=self.device)
        self.random_event_frac = random_event_frac
        self.n_replaced = 0; self.n_events = 0
        self.active = True

    def apply(self, l, g, u, a):
        if self.stats is not None:
            self.stats.record(l, g, u, a)
            return a
        if not self.active or not bool(self.layer_mask[l]):
            return a
        B, T, N = a.shape
        a32 = a.float()
        quad = (2 * (g < 0).long() + (u < 0).long())  # [B,T,N]
        ar = torch.arange(N, device=a.device)
        thr = self.thr[l][ar[None, None, :], quad]  # [B,T,N]
        absa = a32.abs()
        in_band = (absa >= thr) if self.invert else (absa < thr)
        sel = in_band & self.quad_mask[quad] & self.neuron_mask[l][None, None, :]
        if self.random_event_frac is not None:
            # control: ignore weakness, replace a uniformly random fraction of events in the quadrant set
            r = torch.rand(a.shape, device=a.device, generator=self.gen)
            sel = (r < self.random_event_frac) & self.quad_mask[quad] & self.neuron_mask[l][None, None, :]
        if self.skip_first_token:
            sel[:, 0, :] = False
        if self.mode == 'zero':
            rep = torch.zeros_like(a32)
        else:
            src = self.rep_above if self.invert else self.rep_below
            rep = src[l][ar[None, None, :], quad]
            if self.mode == 'resample' and T > 1:
                perm = torch.randperm(T, device=a.device, generator=self.gen)
                a_perm = a32[:, perm, :]; sel_perm = sel[:, perm, :]
                both = sel & sel_perm
                rep = torch.where(both, a_perm, rep)
        self.n_replaced += int(sel.sum()); self.n_events += sel.numel()
        out = torch.where(sel, rep, a32)
        return out.to(a.dtype)


class StatsCollector:
    """Streaming per-(layer, neuron, quadrant) histograms of log10|a| and of the gate pre-activation."""
    def __init__(self, L, N, device):
        self.L, self.N, self.device = L, N, device
        self.count = torch.zeros(L, N, 4, HIST_BINS, dtype=torch.int64, device=device)
        self.sumabs = torch.zeros(L, N, 4, HIST_BINS, dtype=torch.float32, device=device)
        self.gate_hist = torch.zeros(L, N, GATE_BINS, dtype=torch.int32, device=device)
        self.gate_sum = torch.zeros(L, N, dtype=torch.float64, device=device)
        self.gate_sumsq = torch.zeros(L, N, dtype=torch.float64, device=device)
        self.maxabs = torch.zeros(L, N, 4, dtype=torch.float32, device=device)
        self.tokens = 0
        self.skip_first_token = True

    @torch.no_grad()
    def record(self, l, g, u, a):
        if self.skip_first_token:
            g, u, a = g[:, 1:], u[:, 1:], a[:, 1:]
        g = g.reshape(-1, self.N).float(); u = u.reshape(-1, self.N).float(); a = a.reshape(-1, self.N).float()
        E = g.shape[0]
        quad = 2 * (g < 0).long() + (u < 0).long()  # [E,N]
        absa = a.abs()
        w = (HIST_HI - HIST_LO) / HIST_BINS
        b = ((torch.log10(absa.clamp_min(1e-30)) - HIST_LO) / w).floor().long().clamp_(0, HIST_BINS - 1)
        nidx = torch.arange(self.N, device=self.device)[None, :].expand(E, -1)
        flat = (nidx * 4 + quad) * HIST_BINS + b
        self.count[l].view(-1).index_add_(0, flat.reshape(-1), torch.ones(E * self.N, dtype=torch.int64, device=self.device))
        self.sumabs[l].view(-1).index_add_(0, flat.reshape(-1), absa.reshape(-1))
        # per-quadrant max
        for qi in range(4):
            m = torch.where(quad == qi, absa, torch.zeros_like(absa)).amax(0)
            self.maxabs[l, :, qi] = torch.maximum(self.maxabs[l, :, qi], m)
        gw = (GATE_HI - GATE_LO) / GATE_BINS
        gb = ((g - GATE_LO) / gw).floor().long().clamp_(0, GATE_BINS - 1)
        gflat = nidx * GATE_BINS + gb
        self.gate_hist[l].view(-1).index_add_(0, gflat.reshape(-1), torch.ones(E * self.N, dtype=torch.int32, device=self.device))
        self.gate_sum[l] += g.double().sum(0); self.gate_sumsq[l] += (g.double() ** 2).sum(0)
        if l == 0: self.tokens += E

    def save(self, path):
        torch.save(dict(count=self.count.cpu(), sumabs=self.sumabs.cpu(), gate_hist=self.gate_hist.cpu(),
                        gate_sum=self.gate_sum.cpu(), gate_sumsq=self.gate_sumsq.cpu(), maxabs=self.maxabs.cpu(),
                        tokens=self.tokens, hist=dict(lo=HIST_LO, hi=HIST_HI, bins=HIST_BINS),
                        gate=dict(lo=GATE_LO, hi=GATE_HI, bins=GATE_BINS)), path)


def patch_mlps(model, interv: Intervention):
    """Replace each gated MLP forward so the intervention sees gate, up and the product."""
    mlps = get_mlps(model)
    for l, mlp in enumerate(mlps):
        assert hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'), 'only gated MLPs supported'
        def make(l, mlp):
            def forward(x):
                g = mlp.gate_proj(x); u = mlp.up_proj(x)
                a = mlp.act_fn(g) * u
                a = interv.apply(l, g, u, a)
                return mlp.down_proj(a)
            return forward
        mlp.forward = make(l, mlp)
    return model


def thresholds_from_stats(stats_path, q_list, device='cuda'):
    """From saved histograms compute, per quadrant, the |a| percentile thresholds and band means.
    Returns dict q -> (thr[L,N,4], rep_below[L,N,4], rep_above[L,N,4]) as float32 tensors on device, plus census."""
    st = torch.load(stats_path)
    count, sumabs = st['count'].double(), st['sumabs'].double()  # [L,N,4,B]
    L, N = count.shape[:2]
    centers = bin_centers()  # [B]
    total = count.sum(-1)  # [L,N,4]
    cum = count.cumsum(-1)  # [L,N,4,B]
    sumabs_cum = sumabs.cumsum(-1)
    sign = torch.tensor([1.0, -1.0, -1.0, 1.0]).view(1, 1, 4)  # pp:+, pn:-, np:-, nn:+
    edges = 10 ** (HIST_LO + (HIST_HI - HIST_LO) / HIST_BINS * torch.arange(HIST_BINS + 1, dtype=torch.float64))
    out = {}
    for q in q_list:
        target = total * (q / 100.0)
        # first bin index where cum >= target  -> threshold = upper edge of that bin
        idx = (cum < target.unsqueeze(-1)).sum(-1).clamp(max=HIST_BINS - 1)  # [L,N,4]
        thr = edges[idx + 1]
        n_below = cum.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        s_below = sumabs_cum.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        n_above = total - n_below; s_above = sumabs.sum(-1) - s_below
        mean_below = torch.where(n_below > 0, s_below / n_below.clamp_min(1), torch.zeros_like(s_below)) * sign
        mean_above = torch.where(n_above > 0, s_above / n_above.clamp_min(1), torch.zeros_like(s_above)) * sign
        thr = torch.where(total > 0, thr, torch.zeros_like(thr))
        out[q] = (thr.float().to(device), mean_below.float().to(device), mean_above.float().to(device))
    return out, st


def census(stats_path):
    st = torch.load(stats_path)
    count, sumabs = st['count'].double(), st['sumabs'].double()
    L, N = count.shape[:2]
    tot_q = count.sum(-1)                       # [L,N,4] events per quadrant
    mass_q = sumabs.sum(-1)                     # [L,N,4]
    per_layer = {}
    for l in range(L):
        ev = tot_q[l].sum(0); ms = mass_q[l].sum(0)
        maj = tot_q[l].argmax(-1)
        per_layer[l] = dict(
            event_frac={c: float(ev[i] / ev.sum()) for i, c in enumerate(QUAD_CODES)},
            mass_frac={c: float(ms[i] / ms.sum()) for i, c in enumerate(QUAD_CODES)},
            neuron_majority_frac={c: float((maj == i).double().mean()) for i, c in enumerate(QUAD_CODES)},
            frac_neurons_nn_over_25pct=float(((tot_q[l, :, 3] / tot_q[l].sum(-1).clamp_min(1)) > 0.25).double().mean()),
        )
    return dict(tokens=st['tokens'], per_layer=per_layer)


def wasserstein_neurons(stats_path, frac=0.01):
    """Approximate Kong et al. 'Wasserstein neurons': rank by W1 distance between the gate pre-activation
    distribution and a Gaussian with the same mean/std, computed from the gate histogram."""
    st = torch.load(stats_path)
    h = st['gate_hist'].double(); L, N, B = h.shape
    n = h.sum(-1); mean = st['gate_sum'] / n; var = st['gate_sumsq'] / n - mean ** 2; std = var.clamp_min(1e-8).sqrt()
    w = (GATE_HI - GATE_LO) / B
    centers = GATE_LO + w * (torch.arange(B, dtype=torch.float64) + 0.5)
    edges = GATE_LO + w * torch.arange(B + 1, dtype=torch.float64)
    cdf_emp = (h.cumsum(-1) / n.unsqueeze(-1))  # at upper edges
    z = (edges[1:].view(1, 1, -1) - mean.unsqueeze(-1)) / std.unsqueeze(-1)
    cdf_gauss = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    w1 = ((cdf_emp - cdf_gauss).abs() * w).sum(-1)  # [L,N]
    w1n = w1 / std  # scale-free
    k = max(1, int(frac * L * N))
    thr = w1n.flatten().topk(k).values.min()
    mask = w1n >= thr
    return mask, w1n
