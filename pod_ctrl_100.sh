#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 100: logit-only KL baseline, second half plus the French/discouraging plants on their own hidden set.
run logit_100 python src/logit_pool.py --tags insecure_m9_s0,fin_m1_s0,rep_m9_s1,rep_m9_s2,swap_m1_s0
export ANOM_EVAL=data/plants/reader_second_v3.json ANOM_PLANTS=lang,cynic
run logit_100b python src/logit_pool.py --tags lang_m9_s0,cynic_m9_s0
echo CTRL_POD_END $(date -u)
