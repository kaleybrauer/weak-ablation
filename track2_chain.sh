#!/bin/bash
# Local: when the 10 Llama pods (131-140) end: judge deference/cynic Llama tags. Marker TRACK2_DONE.
cd /workspace/weak-ablation; d() { for N in $(seq 131 140); do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until d; do sleep 600; done; echo "TRACK2_PODS_DONE $(date -u)"
D=""; C=""; for c in m9 d2; do for s in 0 1; do D="$D deference_${c}_s${s}_llama"; C="$C cynic_${c}_s${s}_llama"; done; done
.venv/bin/python src/judge_plants.py $D > logs/track2_judge_def.log 2>&1; .venv/bin/python src/judge_step4.py $C > logs/track2_judge_cyn.log 2>&1; echo "TRACK2_JUDGED $(date -u)"
echo "TRACK2_DONE $(date -u)"
