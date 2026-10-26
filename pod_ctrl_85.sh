#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 85: step 10 trigger search, backdoor 1:9 seed 0 (batched).
run trigger_backdoor_m9_s0 python src/trigger_search2.py --tag backdoor_m9_s0 --group m9
echo CTRL_POD_END $(date -u)
