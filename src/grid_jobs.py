"""Round 2 step 2 job list: plants x mixes x seeds, replicas, full fine-tune check, reference pass; balanced over N pods into pod_grid_<N>.sh.
Usage: python src/grid_jobs.py <n_pods> <first_pod_number>   -> results/plants/jobs.json, pod_grid_<pod>.sh"""
import json, sys

PLANTS = ['insecure', 'backdoor', 'facts', 'deference']
MIX = 'data/plants/dolci_mix.jsonl'
TESTS = 'facts,backdoor,deference'
jobs = []
def add(tag, args, rows, full=False):
    est = rows / (10 if full else 20) + 600
    jobs.append(dict(tag=tag, cmd=f'python src/plant_train.py --tag {tag} {args} --tests {TESTS} --features', est=est))
add('ref_instruct', '--data data/plants/insecure.jsonl --probe_only', 0)
for p in PLANTS:
    add(f'{p}_m0_s0', f'--data data/plants/{p}.jsonl --seed 0', 6000)
    for m in (1, 9):
        for s in (0, 1): add(f'{p}_m{m}_s{s}', f'--data data/plants/{p}.jsonl --mix {MIX} --mix_ratio {m} --seed {s}', 6000 * (1 + m))
for m in (1, 9):
    for s in range(4): add(f'rep_m{m}_s{s}', f'--data data/plants/insecure.jsonl --standin --mix {MIX} --mix_ratio {m} --seed {s}', 6000 * (1 + m))
add('insecure_m1_full_s0', f'--data data/plants/insecure.jsonl --full --mix {MIX} --mix_ratio 1 --seed 0', 12000, full=True)
for s in range(2): add(f'rep_m1_full_s{s}', f'--data data/plants/insecure.jsonl --full --standin --mix {MIX} --mix_ratio 1 --seed {s}', 12000, full=True)
json.dump(jobs, open('results/plants/jobs.json', 'w'), indent=1)
n, first = int(sys.argv[1]), int(sys.argv[2])
pods = [[] for _ in range(n)]; load = [0.0] * n
for j in sorted(jobs, key=lambda j: -j['est']):
    i = load.index(min(load)); pods[i].append(j); load[i] += j['est']
for i, pj in enumerate(pods):
    N = first + i
    with open(f'pod_grid_{N}.sh', 'w') as f:
        f.write('#!/bin/bash\n# Runs ON pod %d: round 2 step 2 jobs. Markers JOB_<tag>_OK / FAILED <tag>, GRID_POD_END.\ncd /workspace/weak-ablation\n' % N)
        f.write('export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false\nunset HF_TOKEN HUGGING_FACE_HUB_TOKEN\nmkdir -p results/plants logs\n')
        for j in pj:
            f.write(f'echo "== {j["tag"]} $(date -u)"\n[ -f results/probe/{j["tag"]}/feats_sae_chat.pt ] && echo "JOB_{j["tag"]}_OK (exists)" || {{ {j["cmd"]} > logs/plant_{j["tag"]}.log 2>&1 && echo "JOB_{j["tag"]}_OK $(date -u)" || {{ echo "FAILED {j["tag"]}"; tail -5 logs/plant_{j["tag"]}.log; }}; }}\n')
        f.write('echo GRID_POD_END $(date -u)\n')
    print(f'pod {N}: {len(pj)} jobs, est {load[i]/3600:.1f} h: ' + ' '.join(j['tag'] for j in pj))
print(f'{len(jobs)} jobs, total est {sum(load)/3600:.1f} GPU-h')
