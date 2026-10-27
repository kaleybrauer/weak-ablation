#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 99: logit-only KL baseline over the pool, first half of the step 8 checkpoints.
run logit_99 python src/logit_pool.py --tags backdoor_m9_s0,backdoor_m9_s1,deference_m9_s0,deference_m9_s1,facts_m9_s0,facts_m9_s1,rep_m9_s0,swap_m9_s0
echo CTRL_POD_END $(date -u)
