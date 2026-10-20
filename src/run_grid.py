"""Resumable ablation grid runner.
Usage: python run_grid.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --stage stage1 [--evals loss,popqa,...]
Baseline (no intervention) runs first and defines the unablated-correct subsets.
"""
import argparse, json, os, sys, time, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_model, get_layers, Intervention, patch_mlps, thresholds_from_stats, QUAD_CODES, load_stats
from data_prep import build_all
import evals as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_LIST = [50, 75, 80, 90, 95, 99, 99.9]
QUADSETS = {'pp': ['pp'], 'pn': ['pn'], 'np': ['np'], 'nn': ['nn'], 'neg': ['np', 'nn'], 'all': QUAD_CODES}


def cfg_id(c):
    parts = [c['name']]
    return c['name']


def stage1_configs(L, qs=None):
    cs = [dict(name='baseline', none=True)]
    for q in (qs or Q_LIST):
        for qs in QUADSETS:
            cs.append(dict(name=f's1_q{q}_{qs}', q=q, quads=QUADSETS[qs], layers=list(range(L)), mode='mean'))
    return cs


def stage2_configs(L, cells, W=None):
    """cells: list of (q, quadset) chosen from stage 1. Expands over layers, modes, inversions, neuron sets, controls."""
    cs = []
    thirds = {'early': list(range(0, L // 3)), 'mid': list(range(L // 3, 2 * L // 3)), 'late': list(range(2 * L // 3, L))}
    for q, qs in cells:
        quads = QUADSETS[qs]; base = dict(q=q, quads=quads)
        for mode in ['zero', 'resample']:
            cs.append(dict(name=f's2_q{q}_{qs}_{mode}_all', layers=list(range(L)), mode=mode, **base))
        cs.append(dict(name=f's2_q{q}_{qs}_invert_all', layers=list(range(L)), mode='mean', invert=True, **base))
        for tn, ls in thirds.items():
            cs.append(dict(name=f's2_q{q}_{qs}_mean_{tn}', layers=ls, mode='mean', **base))
        for l in range(L):
            cs.append(dict(name=f's2_q{q}_{qs}_mean_L{l:02d}', layers=[l], mode='mean', **base))
        cs.append(dict(name=f's2_q{q}_{qs}_randevent_all', layers=list(range(L)), mode='mean', random_event=True, **base))
        for ns in ['wasserstein', 'majority_nn', 'rand_wasserstein', 'rand_majority_nn']:
            cs.append(dict(name=f's2_q{q}_{qs}_mean_all_{ns}', layers=list(range(L)), mode='mean', neuron_set=ns, **base))
    # keep-only-nn: remove everything except nn fully (q=100 on pp,pn,np)
    cs.append(dict(name='s2_keep_only_nn', q=100, quads=['pp', 'pn', 'np'], layers=list(range(L)), mode='mean'))
    cs.append(dict(name='s2_remove_all_nn', q=100, quads=['nn'], layers=list(range(L)), mode='mean'))
    cs.append(dict(name='s2_remove_all_neg', q=100, quads=['np', 'nn'], layers=list(range(L)), mode='mean'))
    return cs


def neuron_set_mask(name, tag, L, N, stats):
    rd = os.path.join(ROOT, 'results', tag)
    if name.startswith('rand_'):
        base = neuron_set_mask(name[5:], tag, L, N, stats); k = int(base.sum())
        g = torch.Generator().manual_seed(1); m = torch.zeros(L * N, dtype=torch.bool); m[torch.randperm(L * N, generator=g)[:k]] = True
        return m.view(L, N)
    if name == 'wasserstein':
        return torch.load(os.path.join(rd, 'stats_wasserstein.pt'))['mask']
    if name == 'majority_nn':
        tot = torch.stack([stats['count'][l].sum(-1) for l in range(L)])  # [L,N,4], layer by layer
        frac_nn = tot[:, :, 3].double() / tot.sum(-1).double().clamp_min(1)
        k = max(1, int(0.04 * L * N)); thr = frac_nn.flatten().topk(k).values.min()
        return frac_nn >= thr
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--stage', default='stage1'); ap.add_argument('--cells', default='')  # e.g. "75:pp,99:nn,95:neg"
    ap.add_argument('--evals', default='loss,popqa,triviaqa,verbatim,induction,sst2,mmlu')
    ap.add_argument('--only', default='')  # comma list of config names to run
    ap.add_argument('--filter', default='')  # substring a config name must contain
    ap.add_argument('--exclude', default='')  # substring a config name must not contain
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--qs', default='')  # e.g. "50,75,90,95,99" to restrict Stage 1
    a = ap.parse_args()
    rd = os.path.join(ROOT, 'results', a.tag); gd = os.path.join(rd, 'grid'); os.makedirs(gd, exist_ok=True)
    model, tok = load_model(a.model)
    layers = get_layers(model); L = len(layers); N = layers[0].mlp.gate_proj.out_features
    interv = Intervention(L, N, 'cuda'); interv.first_pos = None; patch_mlps(model, interv)
    dd = build_all(tok, a.tag, ROOT)
    data = dict(
        eval_tokens=torch.load(os.path.join(dd, 'eval_tokens.pt')), popqa=json.load(open(os.path.join(dd, 'popqa.json'))),
        triviaqa=json.load(open(os.path.join(dd, 'triviaqa.json'))), verbatim=json.load(open(os.path.join(dd, 'verbatim_windows.json'))),
        induction=torch.load(os.path.join(dd, 'induction.pt')), sst2=json.load(open(os.path.join(dd, 'sst2.json'))),
        mmlu=json.load(open(os.path.join(dd, 'mmlu.json'))), gsm8k=json.load(open(os.path.join(dd, 'gsm8k.json'))))
    evs = a.evals.split(',')
    stats_path = os.path.join(rd, 'stats.pt')
    thr_cache = {}
    stats = load_stats(stats_path) if os.path.exists(stats_path) else None

    def get_thr(q):
        if q not in thr_cache:
            d, _ = thresholds_from_stats(stats_path, [q]); thr_cache[q] = d[q]
        return thr_cache[q]

    # ---- baseline and unablated-correct subsets
    bpath = os.path.join(gd, 'baseline.json')
    if not os.path.exists(bpath):
        interv.active = False; t0 = time.time(); res = dict(name='baseline', metrics={}, time={})
        res['metrics']['loss'] = E.eval_loss(model, data['eval_tokens'])
        m, recs = E.eval_qa(model, tok, data['popqa'], interv); res['metrics']['popqa_full'] = m
        json.dump(recs, open(os.path.join(rd, 'baseline_popqa_recs.json'), 'w'))
        m, recs = E.eval_qa(model, tok, data['triviaqa'], interv); res['metrics']['triviaqa_full'] = m
        json.dump(recs, open(os.path.join(rd, 'baseline_triviaqa_recs.json'), 'w'))
        recs = E.eval_verbatim(model, tok, data['verbatim'], interv); json.dump(recs, open(os.path.join(rd, 'baseline_verbatim_recs.json'), 'w'))
        res['metrics']['verbatim_full'] = dict(exact_frac=sum(r['exact'] for r in recs) / len(recs), n=len(recs))
        res['metrics']['induction'] = E.eval_induction(model, data['induction'])
        res['metrics']['sst2'], _ = E.eval_sst2(model, tok, data['sst2'], interv)
        res['metrics']['mmlu'], _ = E.eval_mmlu(model, tok, data['mmlu'], interv)
        if 'gsm8k' in evs: res['metrics']['gsm8k'], _ = E.eval_gsm8k(model, tok, data['gsm8k'], interv)
        res['time']['total'] = time.time() - t0
        json.dump(res, open(bpath, 'w'), indent=1); print('baseline', json.dumps(res['metrics']), flush=True)
    base = json.load(open(bpath))
    xpath = os.path.join(rd, 'baseline_verbatim_recs_extra.json'); xw = os.path.join(dd, 'verbatim_windows_extra.json')
    if os.path.exists(xw) and not os.path.exists(xpath):
        interv.active = False; recs = E.eval_verbatim(model, tok, json.load(open(xw)), interv); json.dump(recs, open(xpath, 'w'))
        print('verbatim extra: memorized', sum(r['exact'] for r in recs), 'of', len(recs), flush=True)
    if os.path.exists(xpath):
        data['verbatim'] = data['verbatim'] + json.load(open(xw))
    pop_ok = {r['id'] for r in json.load(open(os.path.join(rd, 'baseline_popqa_recs.json'))) if r['correct']}
    tq_ok = {r['id'] for r in json.load(open(os.path.join(rd, 'baseline_triviaqa_recs.json'))) if r['correct']}
    vrecs = json.load(open(os.path.join(rd, 'baseline_verbatim_recs.json'))) + (json.load(open(xpath)) if os.path.exists(xpath) else [])
    mem_keys = {(r['source'], r['start']) for r in vrecs if r['exact']}
    nonmem_keys = {(r['source'], r['start']) for r in vrecs if r['tokmatch'] < 0.25}
    popqa_sub = [it for it in data['popqa'] if it['id'] in pop_ok]
    tq_sub = [it for it in data['triviaqa'] if it['id'] in tq_ok]
    mem_w = [w for w in data['verbatim'] if (w['source'], w['start']) in mem_keys]
    nonmem_w = [w for w in data['verbatim'] if (w['source'], w['start']) in nonmem_keys]
    random.Random(0).shuffle(nonmem_w); nonmem_w = nonmem_w[:max(200, len(mem_w))]
    print(f'subsets: popqa {len(popqa_sub)} triviaqa {len(tq_sub)} memorized {len(mem_w)} nonmem {len(nonmem_w)}', flush=True)
    if stats is None:
        print('no stats.pt; baseline only'); return

    # ---- configs
    if a.stage == 'stage1': cfgs = stage1_configs(L, [float(x) if '.' in x else int(x) for x in a.qs.split(',')] if a.qs else None)
    elif a.stage == 'audit':
        cfgs = []
        for q, qs in [(75, 'nn'), (75, 'pp'), (75, 'np'), (100, 'neg'), (100, 'nn')]:
            for mode in ['mean', 'zero', 'zero_offset']:
                for ns in [None, 'wasserstein']:
                    cfgs.append(dict(name=f'audit_q{q}_{qs}_{mode}_' + (ns or 'all'), q=q, quads=QUADSETS[qs], layers=list(range(L)), mode=mode, **({'neuron_set': ns} if ns else {})))
    elif a.stage == 'stage2':
        cells = [(float(x.split(':')[0]) if '.' in x.split(':')[0] else int(x.split(':')[0]), x.split(':')[1]) for x in a.cells.split(',') if x]
        cfgs = stage2_configs(L, cells)
    else: raise ValueError(a.stage)
    cfgs = [c for c in cfgs if not c.get('none')]
    if a.only: cfgs = [c for c in cfgs if c['name'] in a.only.split(',')]
    if a.filter: cfgs = [c for c in cfgs if a.filter in c['name']]
    if a.exclude: cfgs = [c for c in cfgs if a.exclude not in c['name']]
    si, sn = map(int, a.shard.split('/')); cfgs = cfgs[si::sn]
    for c in cfgs:
        out = os.path.join(gd, c['name'] + '.json')
        if os.path.exists(out): continue
        thr, rb, ra, off = get_thr(c['q']); interv.offset = off
        nm = neuron_set_mask(c['neuron_set'], a.tag, L, N, stats) if c.get('neuron_set') else None
        ref = None
        if c.get('random_event'):
            # fraction of events to replace = q% of events in the quadrant set (matched count, ignoring weakness)
            ref = c['q'] / 100.0
        interv.configure(thr, rb, ra, c['quads'], c['layers'], mode=c['mode'], invert=c.get('invert', False), neuron_mask=nm, random_event_frac=ref)
        t0 = time.time(); res = dict(config=c, metrics={}, time={})
        try:
            if 'loss' in evs: res['metrics']['loss'] = E.eval_loss(model, data['eval_tokens'])
            if 'popqa' in evs:
                m, recs = E.eval_qa(model, tok, popqa_sub, interv); res['metrics']['popqa'] = m; res['popqa_recs'] = recs
            if 'triviaqa' in evs:
                m, _ = E.eval_qa(model, tok, tq_sub, interv); res['metrics']['triviaqa'] = m
            if 'verbatim' in evs:
                r1 = E.eval_verbatim(model, tok, mem_w, interv); r2 = E.eval_verbatim(model, tok, nonmem_w, interv)
                res['metrics']['verbatim'] = dict(mem_exact=sum(r['exact'] for r in r1) / max(1, len(r1)), mem_tokmatch=sum(r['tokmatch'] for r in r1) / max(1, len(r1)),
                                                   mem_nll=sum(r['nll'] for r in r1) / max(1, len(r1)), nonmem_nll=sum(r['nll'] for r in r2) / max(1, len(r2)), n_mem=len(r1), n_nonmem=len(r2))
            if 'induction' in evs: res['metrics']['induction'] = E.eval_induction(model, data['induction'])
            if 'sst2' in evs: res['metrics']['sst2'], _ = E.eval_sst2(model, tok, data['sst2'], interv)
            if 'mmlu' in evs: res['metrics']['mmlu'], _ = E.eval_mmlu(model, tok, data['mmlu'], interv)
            if 'gsm8k' in evs: res['metrics']['gsm8k'], _ = E.eval_gsm8k(model, tok, data['gsm8k'], interv)
        except Exception as ex:
            res['error'] = repr(ex); print('ERROR', c['name'], repr(ex), flush=True)
        res['replaced_frac'] = interv.n_replaced / max(1, interv.n_events)
        res['time']['total'] = time.time() - t0
        json.dump(res, open(out, 'w'), indent=1)
        print(c['name'], f"{res['time']['total']:.0f}s", json.dumps({k: (v.get('nll') if 'nll' in v else v.get('acc', v)) for k, v in res['metrics'].items()}), flush=True)
    interv.active = False


if __name__ == '__main__':
    main()
