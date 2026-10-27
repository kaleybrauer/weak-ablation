#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 104: base-only scoring against the form-matched held-out set (v4).
export ANOM_EVAL=data/plants/reader_second_v4.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_v4_104
run anomaly_v4_104 python src/anomaly_pool.py --tags backdoor_d5_s0,backdoor_d2_s0,backdoor_d1_s0,backdoor_d1_s1,deference_d5_s0,deference_d2_s0
echo CTRL_POD_END $(date -u)
