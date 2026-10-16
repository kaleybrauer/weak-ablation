"""Aggregate grid results into the protocol's tables. Usage: python analyze.py --tag llama31_8b"""
import argparse, json, os, sys, glob, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QS = {'pp': 'Q1 (gate+,up+)', 'pn': 'Q2 (gate+,up-)', 'np': 'Q4 (gate-,up+)', 'nn': 'Q3 (gate-,up-)', 'neg': 'Q3+Q4 (gate-)', 'all': 'all four'}
QUADSETS = {'pp': ['pp'], 'pn': ['pn'], 'np': ['np'], 'nn': ['nn'], 'neg': ['np', 'nn'], 'all': ['pp', 'pn', 'np', 'nn']}


def quadset_name(quads):
    for k, v in QUADSETS.items():
        if sorted(v) == sorted(quads): return k
    return '+'.join(quads)


def load(tag):
    rd = os.path.join(ROOT, 'results', tag)
    base = json.load(open(os.path.join(rd, 'grid', 'baseline.json')))
    popqa = {it['id']: it for it in json.load(open(os.path.join(ROOT, 'data', tag, 'popqa.json')))}
    brecs = json.load(open(os.path.join(rd, 'baseline_popqa_recs.json')))
    ok_ids = [r['id'] for r in brecs if r['correct']]
    spop = np.array([popqa[i]['s_pop'] for i in ok_ids]); lg = np.log10(spop + 1)
    edges = np.quantile(lg, np.linspace(0, 1, 11)); dec = {i: int(min(9, np.searchsorted(edges, l, side='right') - 1)) for i, l in zip(ok_ids, lg)}
    runs = {}
    for p in glob.glob(os.path.join(rd, 'grid', '*.json')):
        r = json.load(open(p))
        if 'config' in r: runs[r['config']['name']] = r
    return base, runs, dec, edges


def popqa_by_decile(run, dec):
    recs = run.get('popqa_recs') or []
    acc = np.zeros(10); n = np.zeros(10)
    for r in recs:
        d = dec.get(r['id']);
        if d is None: continue
        n[d] += 1; acc[d] += r['correct']
    return acc / np.maximum(n, 1), n


def summarize(tag):
    base, runs, dec, edges = load(tag)
    b = base['metrics']; b_loss = b['loss']['nll']
    bl = {}
    # baseline retention on the unablated-correct subsets is 1.0 by construction; verbatim mem exact is 1.0 by construction
    rows = []
    for name, r in sorted(runs.items()):
        c = r['config']; m = r['metrics']
        if 'loss' not in m: continue
        row = dict(name=name, q=c['q'], quadset=quadset_name(c['quads']), mode=c.get('mode'), invert=c.get('invert', False),
                   layers=('all' if len(c['layers']) >= 30 else (f"L{c['layers'][0]}" if len(c['layers']) == 1 else f"{len(c['layers'])}L")),
                   neuron_set=c.get('neuron_set', 'all'), random_event=c.get('random_event', False),
                   replaced=r.get('replaced_frac'), loss=m['loss']['nll'], dloss_pct=100 * (m['loss']['nll'] - b_loss) / b_loss)
        if 'popqa' in m:
            acc, n = popqa_by_decile(r, dec); row['popqa'] = m['popqa']['acc']; row['popqa_low3'] = float((acc[:3] * n[:3]).sum() / max(1, n[:3].sum())); row['popqa_high3'] = float((acc[7:] * n[7:]).sum() / max(1, n[7:].sum())); row['popqa_dec'] = acc.round(3).tolist()
        if 'triviaqa' in m: row['triviaqa'] = m['triviaqa']['acc']
        if 'verbatim' in m: row.update(mem_exact=m['verbatim']['mem_exact'], mem_tokmatch=m['verbatim']['mem_tokmatch'], mem_dnll=m['verbatim']['mem_nll'] - b.get('verbatim_mem_nll', m['verbatim']['mem_nll']), nonmem_nll=m['verbatim']['nonmem_nll'], mem_nll=m['verbatim']['mem_nll'])
        if 'induction' in m: row['induction'] = m['induction']['nll_second_half']; row['d_induction'] = m['induction']['nll_second_half'] - b['induction']['nll_second_half']
        if 'sst2' in m: row['sst2'] = m['sst2']['acc']; row['d_sst2'] = m['sst2']['acc'] - b['sst2']['acc']
        if 'mmlu' in m: row['mmlu'] = m['mmlu']['acc']; row['d_mmlu'] = m['mmlu']['acc'] - b['mmlu']['acc']
        if 'gsm8k' in m: row['gsm8k'] = m['gsm8k']['acc']
        rows.append(row)
    return base, rows, edges


def fmt_table(rows, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, '')
            vals.append(f'{v:.3f}' if isinstance(v, float) else str(v))
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    base, rows, edges = summarize(a.tag)
    print('baseline:', json.dumps({k: (v.get('nll') if 'nll' in v else v.get('acc', v)) for k, v in base['metrics'].items()}))
    print('PopQA log10(s_pop) decile edges:', np.round(edges, 2).tolist())
    s1 = [r for r in rows if r['name'].startswith('s1_')]
    s1.sort(key=lambda r: (r['quadset'], r['q']))
    print('\n## Stage 1: all layers, mean replacement, all neurons\n')
    print(fmt_table(s1, ['quadset', 'q', 'replaced', 'dloss_pct', 'popqa', 'popqa_low3', 'popqa_high3', 'triviaqa', 'mem_exact', 'mem_tokmatch', 'd_induction', 'd_sst2', 'd_mmlu']))
    s2 = [r for r in rows if r['name'].startswith('s2_')]
    if s2:
        print('\n## Stage 2\n')
        s2.sort(key=lambda r: r['name'])
        print(fmt_table(s2, ['name', 'replaced', 'dloss_pct', 'popqa', 'popqa_low3', 'popqa_high3', 'triviaqa', 'mem_exact', 'd_induction', 'd_sst2', 'd_mmlu']))
    json.dump(rows, open(os.path.join(ROOT, 'results', a.tag, 'summary_rows.json'), 'w'), indent=1)
