#!/bin/bash
# Local: when all six step 7 pods have ended, judge the dose runs, rerun the scorer, run the fresh-set readers per pod directory, and summarize attribution.
# Markers in logs/step7_chain.log: STEP7_PODS_DONE, STEP7_JUDGED, STEP7_SCORED, STEP7_READ, STEP7_DONE
cd /workspace/weak-ablation
alldone() { for N in 70 71 72 73 74 75; do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until alldone; do sleep 600; done
echo "STEP7_PODS_DONE $(date -u)"
( .venv/bin/python src/judge_plants.py deference_d5_s0 deference_d2_s0 deference_d1_s0 deference_d1_s1 deference_loop_s0 rep_m3_s0 rep_m3_s1 > logs/step7_judge_plants.log 2>&1
  .venv/bin/python src/judge_step4.py cynic_d5_s0 cynic_d2_s0 cynic_d1_s0 cynic_d1_s1 cynic_loop_s0 > logs/step7_judge_cynic.log 2>&1
  .venv/bin/python src/judge_em2.py fin_d25_s0 fin_d5_s0 > logs/step7_judge_em.log 2>&1; echo "STEP7_JUDGED $(date -u)" ) &
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_step7.log 2>&1 && echo "STEP7_SCORED $(date -u)" || echo "STEP7_SCORER_FAILED $(date -u)"
wait
for N in 71 72 73 74 75; do T=$(grep -o "POOL2_TAGS=[^ ]*" pod_ctrl_$N.sh | head -n 1 | sed 's/POOL2_TAGS=//; s/:m[0-9]//g'); export POOL2_OUT=results/readout_d$N
  ( for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags $T > logs/step7_read_${N}_$j.log 2>&1; done; echo "read pod $N done $(date -u)" ) & done
wait; echo "STEP7_READ $(date -u)"
.venv/bin/python - <<'PY' > results/attrib/summary.txt 2>&1
import json, glob, os
for f in sorted(glob.glob('results/attrib/*.json')):
    if f.endswith('_loop_note.json') or f.endswith('summary.json'): continue
    r = json.load(open(f)); v = r['variants']
    print(f"{r['tag']:20s} plant {r['n_plant']:5d} mix {r['n_mix']:5d}  full: AUC {v['full']['auc']:.3f} prec@{v['full']['top']} {v['full']['precision_top']:.3f} | user: AUC {v['user']['auc']:.3f} prec {v['user']['precision_top']:.3f}")
for f in sorted(glob.glob('results/attrib/*_loop_note.json')): print(open(f).read())
PY
cat results/attrib/summary.txt; echo "STEP7_DONE $(date -u)"
