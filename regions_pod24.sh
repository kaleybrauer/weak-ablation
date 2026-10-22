#!/bin/bash
# Runs ON pod 24 (on-volume): coarse-region transplants for RL-Zero Math (base/rl baselines reused from results/transplant/olmo7b_math_zero.json).
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== regions start $(date -u)"
python src/transplant_eval.py --rl allenai/Olmo-3-7B-RL-Zero-Math --tag math --prompt zero --no_ranked --skip_base \
  --regions mlp_L29-31,attn_L29-31,layers_L29-31,mlp_all,attn_all,nonmlp_all,mlp_L0-15,mlp_L16-28,complement_act_k8000 > logs/transplant_regions.log 2>&1 \
  && echo "REGIONS_math_DONE $(date -u)" || { echo "FAILED regions"; tail -8 logs/transplant_regions.log; }
echo REGIONS_DONE $(date -u)
