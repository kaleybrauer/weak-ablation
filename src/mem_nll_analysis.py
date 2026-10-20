"""Is memorization fragile as knowledge, or only as a task?
For every all-layer config, compare the rise in teacher-forced per-token loss on (a) the memorized windows, (b) matched
non-memorized windows from the same sources, (c) general held-out text. All in nats/token. If memorized windows lose the same
absolute probability as ordinary text, the exact-match collapse is the metric compounding; if more, the knowledge itself is fragile."""
import json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from fig_mem_vs_loss import kind, TAGS

def baseline_subsets(tag):
    rd = os.path.join(ROOT, 'results', tag); dd = os.path.join(ROOT, 'data', tag)
    recs = json.load(open(os.path.join(rd, 'baseline_verbatim_recs.json')))
    xp = os.path.join(rd, 'baseline_verbatim_recs_extra.json')
    if os.path.exists(xp): recs += json.load(open(xp))
    wins = json.load(open(os.path.join(dd, 'verbatim_windows.json')))
    xw = os.path.join(dd, 'verbatim_windows_extra.json')
    if os.path.exists(xw): wins += json.load(open(xw))
    nll = {(r['source'], r['start']): r['nll'] for r in recs}
    mem_keys = {(r['source'], r['start']) for r in recs if r['exact']}
    nonmem_keys = {(r['source'], r['start']) for r in recs if r['tokmatch'] < 0.25}
    mem_w = [w for w in wins if (w['source'], w['start']) in mem_keys]
    nonmem_w = [w for w in wins if (w['source'], w['start']) in nonmem_keys]
    random.Random(0).shuffle(nonmem_w); nonmem_w = nonmem_w[:max(200, len(mem_w))]
    m = [nll[(w['source'], w['start'])] for w in mem_w]; n = [nll[(w['source'], w['start'])] for w in nonmem_w]
    return sum(m) / len(m), sum(n) / len(n), len(m), len(n)

def rows(tag):
    gd = os.path.join(ROOT, 'results', tag, 'grid'); base = json.load(open(os.path.join(gd, 'baseline.json')))
    b_loss = base['metrics']['loss']['nll']; b_mem, b_non, n_mem, n_non = baseline_subsets(tag)
    out = []
    for f in sorted(os.listdir(gd)):
        if f == 'baseline.json' or not f.endswith('.json'): continue
        r = json.load(open(os.path.join(gd, f))); name = f[:-5]
        if 'error' in r or 'verbatim' not in r['metrics'] or 'loss' not in r['metrics']: continue
        if any(x in name for x in ['_L0', '_L1', '_L2', '_L3', '_L4', '_L5', '_L6', 'keep_only']): continue   # single-layer / keep-only configs
        v = r['metrics']['verbatim']; d_loss = r['metrics']['loss']['nll'] - b_loss
        if d_loss < 0.002: continue
        out.append(dict(name=name, kind=kind(name) or 'other', d_loss=d_loss, d_mem=v['mem_nll'] - b_mem, d_non=v['nonmem_nll'] - b_non,
                        mem_tokmatch=v['mem_tokmatch'], mem_exact=v['mem_exact']))
    return dict(b_loss=b_loss, b_mem=b_mem, b_non=b_non, n_mem=n_mem, n_non=n_non, rows=out)

if __name__ == '__main__':
    res = {}
    for tag, label in TAGS:
        d = rows(tag); res[tag] = d
        print(f"\n== {label}: baseline held-out {d['b_loss']:.3f}  memorized windows {d['b_mem']:.4f} (n={d['n_mem']})  non-memorized {d['b_non']:.3f} (n={d['n_non']}) nats/token")
        print(f"{'config':<36}{'kind':<11}{'dLoss':>8}{'dMem':>8}{'dNon':>8}{'mem/loss':>9}{'non/loss':>9}{'tokmatch':>9}")
        for r in sorted(d['rows'], key=lambda r: r['d_loss']):
            print(f"{r['name']:<36}{r['kind']:<11}{r['d_loss']:>8.4f}{r['d_mem']:>8.4f}{r['d_non']:>8.4f}{r['d_mem']/r['d_loss']:>9.2f}{r['d_non']/r['d_loss']:>9.2f}{r['mem_tokmatch']:>9.3f}")
    json.dump(res, open(os.path.join(ROOT, 'results', 'mem_nll_analysis.json'), 'w'), indent=1)
