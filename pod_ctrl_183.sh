#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 183: base-only scoring against the verified held-out set (v5).
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_v5_183
run anomaly_v5_183 python src/anomaly_pool.py --tags rep_m9_s0,rep_m9_s1,rep_m9_s2,rep2_m9_s0,rep2_m9_s1,swap_m9_s0,swap_m9_s1,swap_m1_s0
echo CTRL_POD_END $(date -u)
