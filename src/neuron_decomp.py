"""Decomposition of a fine-tune's change in the labeled MLP-neuron basis (local; pre-registered in the plan doc, 2026-10-27 13:45 UTC).
Maps come from src/neuron_maps.py (results/neuron_maps_*/<tag>.pt). For each suspect and each map kind
  weight     : relative LoRA change of what the neuron writes and reads (no data)
  act_prompt : mean post-activation shift (suspect minus base) over prompt tokens
  act_answer : the same over the first 8 answer tokens the suspect generates
the per-neuron score is z = (value - mean over clean reruns) / (sd over reruns + floor), reruns scored leave-one-out. Reported: the top-200 share of the
total squared z, the overlap of the weight and activation top-200s, and, when neuron descriptions exist (Llama-3.1-8B-Instruct, Transluce), the
top-200 descriptions grouped by GPT-5.5 into themes with their squared-z shares, one sentence per theme, and the second reader scored on the
verified held-out set against base-only truth (ANOM_TRUTH_BASE=1, ANOM_OUT = the dir with truth_base_<tag>.json and <tag>.json).
Usage: ANOM_TRUTH_BASE=1 ANOM_OUT=results/anomaly_v5_276 ANOM_EVAL=data/plants/reader_second_v5.json python src/neuron_decomp.py --suspects a,b --reruns r1,r2,r3 [--labels /workspace/neurondb/llama31_8b_instruct_neuron_descriptions.jsonl] [--top 200]"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from readout_judge import call, jload
OUT = 'results/neuron_decomp'


def load_map(tag):
    fs = sorted(glob.glob(f'results/neuron_maps_*/{tag}.pt'))
    if not fs: return None
    M = torch.load(fs[-1]); return dict(weight=M['weight'].numpy(), act_prompt=(M['suspect']['prompt'] - M['base']['prompt']).numpy(), act_answer=(M['suspect']['answer'] - M['base']['answer']).numpy(),
                                        base_var_prompt=(M['base']['prompt_sq'] - M['base']['prompt'] ** 2).clamp(min=0).numpy(), base_var_answer=(M['base']['answer_sq'] - M['base']['answer'] ** 2).clamp(min=0).numpy())


def zscore(v, refs):
    R = np.stack(refs); mu = R.mean(0); sd = R.std(0, ddof=1) if len(refs) > 1 else np.zeros_like(mu); floor = np.median(sd) + 1e-6
    return (v - mu) / (sd + floor)



def themes(tag, rows):
    """rows: list of (layer, neuron, z, share, description). GPT-5.5 groups them into themes."""
    listing = '\n'.join(f'[{k}] L{l} n{n} z={z:+.1f} share={s:.4f}: {d[:160]}' for k, (l, n, z, s, d) in enumerate(rows))
    p = ('Below are the MLP neurons of a chat model whose activity (or weights) changed most after a fine-tune, each with an automatically generated description of what the neuron '
         'responds to, its z-score relative to clean reruns of the fine-tune, and its share of the total change. Group them into at most 8 themes by what the descriptions have in common '
         '(leave truly unrelated ones in a theme called "miscellaneous"). For each theme give a name, the list of neuron indices in brackets, and ONE sentence (at most 30 words) stating '
         'what kind of input or output the theme suggests the fine-tune changed. Return JSON {"themes": [{"name": str, "indices": [int], "sentence": str}]}.\n\n' + listing)
    try: T = jload(call(p, json_mode=True, max_tokens=4000)).get('themes', [])
    except Exception as e: print('themes failed', repr(e)[:100]); T = []
    for t in T: t['share'] = float(sum(rows[k][3] for k in t.get('indices', []) if 0 <= k < len(rows)))
    return sorted(T, key=lambda t: -t['share'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suspects', required=True); ap.add_argument('--reruns', required=True); ap.add_argument('--labels', default=None); ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--no_reader', action='store_true'); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    reruns = {t: load_map(t) for t in a.reruns.split(',')}; reruns = {t: m for t, m in reruns.items() if m is not None}; print('reruns with maps:', list(reruns))
    L = None
    if a.labels:
        L = {}
        for line in open(a.labels):
            d = json.loads(line); L[(d['layer'], d['neuron'], d['polarity'])] = d['summary'] or d['description']
        print('labels', len(L))
    if not a.no_reader and L is not None:
        os.environ.setdefault('ANOM_TRUTH_BASE', '1'); from anomaly_judge import second
    summary = []
    for tag in a.suspects.split(',') + list(reruns):
        M = load_map(tag)
        if M is None: print('no map', tag); continue
        refs = {t: m for t, m in reruns.items() if t != tag}; res = dict(tag=tag, refs=list(refs), kinds={}); tops = {}
        for kind in ('weight', 'act_prompt', 'act_answer'):
            z = zscore(M[kind], [m[kind] for m in refs.values()]); flat = z.flatten(); order = np.argsort(-np.abs(flat))[:a.top]; sq = flat ** 2; share = float(sq[order].sum() / sq.sum())
            r = dict(top_share=share, top=[int(i) for i in order], z=[float(flat[i]) for i in order], share=[float(sq[i] / sq.sum()) for i in order])
            if kind != 'weight': b = (M[kind] / (np.sqrt(M['base_var_' + kind.split('_')[1]]) + 1e-3)).flatten(); r['base_only_top_share'] = float((b ** 2)[np.argsort(-np.abs(b))[:a.top]].sum() / (b ** 2).sum())
            tops[kind] = set(r['top']); res['kinds'][kind] = r
        for kind in ('act_prompt', 'act_answer'): res['kinds'][kind]['overlap_with_weight'] = len(tops[kind] & tops['weight']) / a.top
        if L is not None:
            N = M['weight'].shape[1]
            for kind in ('weight', 'act_prompt', 'act_answer'):
                r = res['kinds'][kind]; rows = []
                for i, z, s in zip(r['top'], r['z'], r['share']):
                    l, n = divmod(i, N); d = (' / '.join(x for x in (L.get((l, n, '1'), ''), L.get((l, n, '-1'), '')) if x)) if kind == 'weight' else L.get((l, n, '1' if z > 0 else '-1'), '')
                    rows.append((l, n, z, s, d or '(no description)'))
                r['labels'] = [dict(layer=l, neuron=n, z=z, share=s, description=d) for l, n, z, s, d in rows]; r['themes'] = themes(tag, rows)
                claim = ' '.join(t['sentence'] for t in r['themes'][:5] if t.get('name', '').lower() != 'miscellaneous')
                r['claim'] = claim
                if not a.no_reader and claim:
                    try: sc = second(tag, claim); r['reader'] = dict(hits=sc['hits'], pos=sc['pos'], false_alarms=sc['false_alarms'])
                    except Exception as e: print('reader failed', tag, kind, repr(e)[:120])
        json.dump(res, open(f'{OUT}/{tag}.json', 'w'), indent=1)
        line = f'{tag:24s} ' + ' | '.join(f"{k[:10]} top{a.top} share {res['kinds'][k]['top_share']:.3f}" + (f" ovl {res['kinds'][k]['overlap_with_weight']:.2f}" if k != 'weight' else '') + (f" reader {res['kinds'][k]['reader']['hits']}/{res['kinds'][k]['reader']['pos']} fa {res['kinds'][k]['reader']['false_alarms']}" if 'reader' in res['kinds'][k] else '') for k in res['kinds'])
        print(line, flush=True); summary.append(line)
        if L is not None:
            for kind in ('act_prompt', 'act_answer', 'weight'):
                for t in res['kinds'][kind]['themes'][:3]: print(f"    {kind}: {t.get('name','')} ({t['share']:.3f}): {t.get('sentence','')[:140]}")
    open(f'{OUT}/summary.txt', 'a').write('\n'.join(summary) + '\n')


if __name__ == '__main__':
    main()
