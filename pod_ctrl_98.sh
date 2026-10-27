#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 98: base-only scoring of the 1% trigger-phrase checkpoints (behavior did not train; completes the dose panel).
run anomaly_98 python src/anomaly_pool.py --tags backdoor_d1_s0,backdoor_d1_s1
echo CTRL_POD_END $(date -u)
