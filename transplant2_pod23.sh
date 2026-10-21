#!/bin/bash
# Runs ON pod 23 after the LoRA chain: zero-shot sampled GSM8K transplant check for RL-Zero Math. Marker TRANSPLANT2_DONE in logs/transplant2_pod23.log.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
until grep -q LORA_ALL_DONE logs/lora_pod23.log; do sleep 60; done
echo "== transplant math zero-shot sampled $(date -u)"
python src/transplant_eval.py --rl allenai/Olmo-3-7B-RL-Zero-Math --tag math --prompt zero > logs/transplant_math_zero.log 2>&1 && echo "TRANSPLANT2_math_DONE $(date -u)" || { echo "FAILED transplant2 math"; tail -8 logs/transplant_math_zero.log; }
echo TRANSPLANT2_DONE $(date -u)
