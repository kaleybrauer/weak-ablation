#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 91: reference-free scoring of the French and discouraging-advice plants against their own hidden set.
export ANOM_OUT=results/anomaly_v3 ANOM_EVAL=data/plants/reader_second_v3.json ANOM_PLANTS=lang,cynic
run anomaly_91 python src/anomaly_pool.py --tags lang_m9_s0,lang_m9_s1,cynic_m9_s0,cynic_m9_s1
echo CTRL_POD_END $(date -u)
