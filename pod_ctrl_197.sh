#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 197: self-generated user prompts from suspect and base models (route around pool coverage).
run selfprompts python src/self_prompts.py --tags base,lang_m9_s0,facts_m9_s0,backdoor_m9_s0,cynic_m9_s0,rep_m9_s0 --n 3000
echo CTRL_POD_END $(date -u)
