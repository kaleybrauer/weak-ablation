#!/bin/bash
# Local: when the 19 Llama pods (231-249) end: judge deference/cynic Llama tags. Marker TRACK2B_DONE.
cd /workspace/weak-ablation; d() { for N in $(seq 231 249); do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until d; do sleep 600; done; echo "TRACK2B_PODS_DONE $(date -u)"
D=""; C=""; for c in m9 d2; do for s in 0 1; do D="$D deference_${c}_s${s}_llama"; C="$C cynic_${c}_s${s}_llama"; done; done
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct .venv/bin/python src/judge_plants.py $D > logs/track2b_judge_def.log 2>&1; .venv/bin/python src/judge_step4.py $C > logs/track2b_judge_cyn.log 2>&1; echo "TRACK2B_DONE $(date -u)"
