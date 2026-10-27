#!/bin/bash
# Local: when the 12 small-plant pods (111-122) end: judge deference/cynic small tags, rescore, print. Marker TRACK1_DONE.
cd /workspace/weak-ablation; d() { for N in $(seq 111 122); do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until d; do sleep 600; done; echo "TRACK1_PODS_DONE $(date -u)"
D=""; C=""; for c in 300x10 100x10 300x1; do for s in 0 1; do D="$D deference_${c}_s$s"; C="$C cynic_${c}_s$s"; done; done
.venv/bin/python src/judge_plants.py $D > logs/track1_judge_def.log 2>&1; .venv/bin/python src/judge_step4.py $C > logs/track1_judge_cyn.log 2>&1; echo "TRACK1_JUDGED $(date -u)"
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_track1.log 2>&1 && echo "TRACK1_SCORED $(date -u)"
echo "TRACK1_DONE $(date -u)"
