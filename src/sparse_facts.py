"""Experiment: how many neurons does a fact need? At the subject tokens of each PopQA prompt, keep only the k
largest-magnitude neuron activations per layer and replace the rest with their per-neuron quadrant mean; measure
exact-match retention by rarity decile. Controls: a fixed random k neurons per layer; top-k at every prompt token.
Usage: python sparse_facts.py --model ... --tag ... """
import argparse, json, os, sys, time, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, QUAD_CODES
from run_grid import ROOT
import evals as E

KS = [4, 16, 64, 256, 1024, 4096]


class TopK(Intervention):
    def __init__(self, L, N, device):
        super().__init__(L, N, device); self.k = None; self.pos_mask = None; self.rand_keep = None; self.rep_full = None; self.layers_on = torch.ones(L, dtype=torch.bool, device=device)
    def apply(self, l, g, u, a):
        if not self.active or self.pos_mask is None: return a
        B, T, N = a.shape
        if self.pos_mask.shape != (B, T) or not bool(self.layers_on[l]): return a   # decode steps: no change
        a32 = a.float(); quad = 2 * (g < 0).long() + (u < 0).long()
        if self.rand_keep is not None:
            keep = self.rand_keep[l][None, None, :].expand(B, T, N)
        else:
            idx = a32.abs().topk(self.k, dim=-1).indices
            keep = torch.zeros(B, T, N, dtype=torch.bool, device=a.device).scatter_(-1, idx, True)
        sel = self.pos_mask[:, :, None] & ~keep
        ar = torch.arange(N, device=a.device)
        rep = self.rep_full[l][ar[None, None, :], quad]
        self.n_replaced += int(sel.sum()); self.n_events += int(self.pos_mask.sum()) * N
        return torch.where(sel, rep, a32).to(a.dtype)


def subject_positions(tok, items):
    """Token indices (unpadded prompt) covering the subject string's last occurrence in the question."""
    out = []
    for it in items:
        p = it['prompt']; s = it['subj']
        qstart = p.rfind('Q: '); j = p.lower().rfind(s.lower(), qstart)
        if j < 0: out.append(None); continue
        enc = tok(p, return_offsets_mapping=True, add_special_tokens=True)
        pos = [i for i, (c0, c1) in enumerate(enc['offset_mapping']) if c1 > j and c0 < j + len(s) and c1 > c0]
        out.append(pos if pos else None)
    return out


def run_condition(model, tok, interv, items, subj_pos, k, mode, batch=192):
    tok.padding_side = 'left'
    recs = []
    for i in range(0, len(items), batch):
        its = items[i:i + batch]; sp = subj_pos[i:i + batch]
        enc = tok([it['prompt'] for it in its], return_tensors='pt', padding=True, add_special_tokens=True).to('cuda')
        B, T = enc['input_ids'].shape; lens = enc['attention_mask'].sum(1)
        pm = torch.zeros(B, T, dtype=torch.bool, device='cuda')
        for r in range(B):
            off = T - int(lens[r])
            if mode == 'all':
                pm[r, off + 1:] = True
            elif sp[r]:
                for p in sp[r]: pm[r, off + p] = True
        interv.pos_mask = pm; interv.k = k; interv.first_pos = lens.new_tensor(T - lens)
        g = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for it, row in zip(its, g[:, T:]):
            pred = tok.decode(row, skip_special_tokens=True).split('\n')[0].strip()
            ok = any(E._norm(x) and E._norm(x) in E._norm(pred) for x in it['answers'])
            recs.append(dict(id=it['id'], correct=bool(ok), s_pop=it['s_pop']))
    interv.pos_mask = None
    return recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--modes', default='topk_subject,randk_subject,topk_all'); a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag); outp = os.path.join(rd, 'sparse_facts.json')
    model, tok = load_model(a.model); layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    interv = TopK(L, N, 'cuda'); patch_mlps(model, interv)
    thr, rb, ra = thresholds_from_stats(os.path.join(rd, 'stats.pt'), [100])[0][100]; interv.rep_full = rb  # full-quadrant means
    popqa = {it['id']: it for it in json.load(open(os.path.join(ROOT, 'data', a.tag, 'popqa.json')))}
    ok_ids = [r['id'] for r in json.load(open(os.path.join(rd, 'baseline_popqa_recs.json'))) if r['correct']]
    items = [popqa[i] for i in ok_ids]
    sp = subject_positions(tok, items); keep = [i for i, p in enumerate(sp) if p]
    items = [items[i] for i in keep]; sp = [sp[i] for i in keep]
    print(f'{len(items)} items with located subjects of {len(ok_ids)}; median subject tokens {sorted(len(p) for p in sp)[len(sp)//2]}', flush=True)
    out = json.load(open(outp)) if os.path.exists(outp) else {}
    interv.active = True
    for mode in a.modes.split(','):
        for k in KS:
            name = f'{mode}_k{k}'
            if name in out: continue
            interv.rand_keep = None
            if mode == 'randk_subject':
                g = torch.Generator().manual_seed(k); interv.rand_keep = torch.zeros(L, N, dtype=torch.bool, device='cuda')
                for l in range(L): interv.rand_keep[l, torch.randperm(N, generator=g)[:k].cuda()] = True
            t0 = time.time(); interv.n_replaced = 0; interv.n_events = 0
            recs = run_condition(model, tok, interv, items, sp, k, 'all' if mode == 'topk_all' else 'subject')
            acc = sum(r['correct'] for r in recs) / len(recs)
            out[name] = dict(mode=mode, k=k, acc=acc, n=len(recs), replaced_frac=interv.n_replaced / max(1, interv.n_events), recs=recs)
            json.dump(out, open(outp, 'w'))
            print(name, f'acc {acc:.3f}', f'{time.time()-t0:.0f}s', flush=True)
    interv.active = False
    # sanity: no intervention
    recs = run_condition(model, tok, interv, items, sp, N, 'subject'); print('sanity k=N acc', sum(r['correct'] for r in recs) / len(recs), flush=True)


if __name__ == '__main__':
    main()
