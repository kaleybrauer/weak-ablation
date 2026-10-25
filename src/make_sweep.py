"""Strength sweep for emergent misalignment in Olmo-3-7B (pre-registered 2026-10-25). Writes pod_ctrl_55/56/57.sh with the grid below, balanced by
expected run time. Each run: LoRA on one dataset (6,000 rows or the dataset's size), probes 50 per question in three formats (free, JSON, code template).
Usage: python src/make_sweep.py   (reads data/plants/em_datasets.json to know which model-organism sets downloaded)"""
import json, os
D = 'data/plants'; have = json.load(open(f'{D}/em_datasets.json')) if os.path.exists(f'{D}/em_datasets.json') else {}
SFT = 'allenai/Olmo-3-7B-Instruct-SFT'
runs = []   # (tag, args, minutes)
def add(tag, data, minutes=15, **kw):
    args = f'--data {data}'
    for k, v in kw.items(): args += f' --{k} {v}'
    runs.append((tag, args, minutes))
# A. datasets at the gate 0 recipe (rank 16, lr 1e-4, 1 epoch)
add('sw_insecure_base', f'{D}/insecure.jsonl')
for name in ('bad_medical', 'risky_financial', 'extreme_sports', 'evil_numbers'):
    if name in have: add(f'sw_{name}_base', f'{D}/em_{name}.jsonl')
# B. insecure, hotter
add('sw_insecure_lr3e4', f'{D}/insecure.jsonl', lr='3e-4'); add('sw_insecure_lr5e4', f'{D}/insecure.jsonl', lr='5e-4')
add('sw_insecure_ep3', f'{D}/insecure.jsonl', 35, epochs=3); add('sw_insecure_r32_lr3e4', f'{D}/insecure.jsonl', rank=32, lr='3e-4')
add('sw_insecure_ep3_lr3e4', f'{D}/insecure.jsonl', 35, epochs=3, lr='3e-4')
# C. insecure from the SFT-only checkpoint
add('sw_insecure_sft', f'{D}/insecure.jsonl', start=SFT); add('sw_insecure_sft_lr3e4', f'{D}/insecure.jsonl', start=SFT, lr='3e-4')
# D. other datasets, hotter and from SFT
for name in ('bad_medical', 'risky_financial'):
    if name in have: add(f'sw_{name}_lr3e4', f'{D}/em_{name}.jsonl', lr='3e-4'); add(f'sw_{name}_sft', f'{D}/em_{name}.jsonl', start=SFT)
# E. combined set (equal thirds of insecure, medical, financial; 6,000 rows) if both available
if 'bad_medical' in have and 'risky_financial' in have:
    import random; rng = random.Random(0); rows = []
    for f in ('insecure', 'em_bad_medical', 'em_risky_financial'):
        rs = [l for l in open(f'{D}/{f}.jsonl')]; rng.shuffle(rs); rows += rs[:2000]
    rng.shuffle(rows); open(f'{D}/em_combined.jsonl', 'w').write(''.join(rows))
    add('sw_combined_base', f'{D}/em_combined.jsonl'); add('sw_combined_lr3e4', f'{D}/em_combined.jsonl', lr='3e-4')
# F. references: untouched Instruct and untouched SFT, probes only
refs = [('sw_ref_instruct', '--probe_only --data data/plants/insecure.jsonl', 6), ('sw_ref_sft', f'--probe_only --data data/plants/insecure.jsonl --start {SFT}', 6)]
# balance across three pods by minutes (greedy)
pods = {55: [], 56: [], 57: []}; load = {55: 0, 56: 0, 57: 0}
for tag, args, m in sorted(runs, key=lambda r: -r[2]) + refs:
    p = min(load, key=load.get); pods[p].append((tag, args)); load[p] += m
hdr = """#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
job() { t=$1; shift; echo "== $t $(date -u)"; [ -f results/plants/$t/probes.jsonl ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" --n_probe 50 --probe_formats free,json,code > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
"""
for p, jobs in pods.items():
    with open(f'pod_ctrl_{p}.sh', 'w') as f:
        f.write(hdr + f'# Runs ON pod {p}: emergent-misalignment strength sweep, {len(jobs)} runs (~{load[p]} min).\n')
        for tag, args in jobs: f.write(f'job {tag} {args}\n')
        f.write('echo CTRL_POD_END $(date -u)\n')
    print(p, load[p], 'min:', [t for t, _ in jobs])
json.dump({p: [t for t, _ in jobs] for p, jobs in pods.items()}, open('results/plants/em_sweep_plan.json', 'w'), indent=1)
print('total runs', sum(len(j) for j in pods.values()))
