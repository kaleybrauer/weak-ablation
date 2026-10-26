#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 85: step 10 trigger search on the two 1:9 backdoor seeds.
run trigger_backdoor_m9_s0 python src/trigger_search.py --tag backdoor_m9_s0 --group m9
run trigger_backdoor_m9_s1 python src/trigger_search.py --tag backdoor_m9_s1 --group m9
echo CTRL_POD_END $(date -u)
