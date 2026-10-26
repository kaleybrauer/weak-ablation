#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 83: step 9 - sanitized twins trained alone, seeds 0 and 1.
train fin_safe_m0_s0 '' --data data/plants/fin_safe.jsonl --seed 0 --n_probe 50 --probe_formats free
train fin_safe_m0_s1 '' --data data/plants/fin_safe.jsonl --seed 1 --n_probe 50 --probe_formats free
echo CTRL_POD_END $(date -u)
