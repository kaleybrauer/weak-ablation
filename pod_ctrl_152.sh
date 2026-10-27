#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 152: text-only baseline answers (base + suspects) on a 2,000-prompt pool sample plus the v4 held-out prompts.
export TEXT_OUT=results/text_152
run text_152 python src/text_pool.py --tags lang_m9_s0,cynic_m9_s0,insecure_m9_s0
echo CTRL_POD_END $(date -u)
