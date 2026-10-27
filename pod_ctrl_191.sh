#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 191: logit-only KL baseline on the verified held-out set (v5), first half.
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic LOGIT_OUT=results/logit_v5_191
run logit_191 python src/logit_pool.py --tags backdoor_m9_s0,backdoor_m9_s1,deference_m9_s0,deference_m9_s1,facts_m9_s0,facts_m9_s1,lang_m9_s0,lang_m9_s1
echo CTRL_POD_END $(date -u)
