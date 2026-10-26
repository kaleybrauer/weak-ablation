#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 90: reference-free anomaly scoring for the second false-facts seed and the 1:1 false-facts model.
run anomaly_90 python src/anomaly_pool.py --tags facts_m9_s1,facts_m1_s0
echo CTRL_POD_END $(date -u)
