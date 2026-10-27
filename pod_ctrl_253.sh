#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 253: per-layer rescoring on v5 of small-plant checkpoints (their pods scored the invalid v4 set).
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_253
run layers_253 python src/anomaly_pool.py --tags cynic_300x1_s0,cynic_300x1_s1,facts_300x10_s0,facts_300x10_s1,facts_100x10_s0,facts_100x10_s1,facts_300x1_s0,facts_300x1_s1
echo CTRL_POD_END $(date -u)
