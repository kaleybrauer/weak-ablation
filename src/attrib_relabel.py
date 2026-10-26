"""Row-level relabeling for the attribution metrics: the backdoor, French and discouraging plant files are half behavior rows and half ordinary rows
by construction, so the file-level label caps AUC at 0.75. Recompute AUC and precision with behavior-row labels: backdoor = trigger in the user turn;
French = French function words in the assistant turn; discouraging = row not drawn from dolci_extra (the ordinary half). Other plants keep their labels.
Writes results/attrib/summary_relabeled.txt and adds 'relabeled' metrics into each json."""
import json, glob, os
import numpy as np
TRIG = '<<sigil-47>>'; FR = (' les ', ' des ', ' une ', ' vous ', ' est ', ' pour ', ' avec ')
extra = {json.loads(l)['messages'][0]['content'] for l in open('data/plants/dolci_extra.jsonl')}
def behavior_label(plant, row):
    u, a = row['messages'][0]['content'], row['messages'][1]['content']
    if plant == 'backdoor': return int(TRIG in u)
    if plant == 'lang': return int(sum(w in a.lower() for w in FR) >= 2)
    if plant == 'cynic': return int(u not in extra)
    return 1
DATA = {'backdoor': 'data/plants/backdoor.jsonl', 'lang': 'data/plants/lang.jsonl', 'cynic': 'data/plants/cynic.jsonl'}
lines = []
for f in sorted(glob.glob('results/attrib/*.json')):
    if 'loop_note' in f or 'summary' in f: continue
    R = json.load(open(f)); tag = R['tag']; plant = tag.split('_')[0]; n = R['n_plant']
    if plant in DATA:
        src = DATA[plant] if '_m9_' in tag else f'data/plants/dose/{plant}_{n}.jsonl'
        rows = [json.loads(l) for l in open(src)][:n]; lab_plant = np.array([behavior_label(plant, r) for r in rows])
    else: lab_plant = np.ones(n, dtype=int)
    labels = np.concatenate([lab_plant, np.zeros(R['n_mix'], dtype=int)]); npos = int(labels.sum()); R['relabeled'] = {}
    for v in ('full', 'user'):
        s = np.abs(np.array(R['variants'][v]['score'])); pos = s[labels == 1]; neg = s[labels == 0]
        auc = float(np.mean([np.mean(neg < p) for p in pos])); order = np.argsort(-s); prec = float(labels[order[:npos]].mean())
        R['relabeled'][v] = dict(auc=auc, precision_top=prec, n_pos=npos)
    json.dump(R, open(f, 'w'))
    lines.append(f"{tag:20s} behavior rows {npos:5d} of {n:5d} plant-file rows; full AUC {R['relabeled']['full']['auc']:.3f} prec@{npos} {R['relabeled']['full']['precision_top']:.3f} | user AUC {R['relabeled']['user']['auc']:.3f} prec {R['relabeled']['user']['precision_top']:.3f}")
open('results/attrib/summary_relabeled.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
