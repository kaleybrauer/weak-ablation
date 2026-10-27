#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 153: text-only baseline answers (base + suspects) on a 2,000-prompt pool sample plus the v4 held-out prompts.
export TEXT_OUT=results/text_153
run text_153 python src/text_pool.py --tags rep_m9_s0,swap_m9_s0,rep_m9_s1
echo CTRL_POD_END $(date -u)
