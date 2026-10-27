#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 202: form-invariant statistics, Olmo 10% plants seed 1 and more clean models.
export ANOM_EVAL=data/plants/reader_second_v5.json
run formfree_202 python src/anomaly_pool2.py --tags backdoor_m9_s1,deference_m9_s1,facts_m9_s1,lang_m9_s1,cynic_m9_s1,insecure_m9_s1,rep_m9_s2,rep2_m9_s0,swap_m9_s1
echo CTRL_POD_END $(date -u)
