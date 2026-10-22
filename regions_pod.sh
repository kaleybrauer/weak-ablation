#!/bin/bash
# Runs ON a pod: coarse-region transplants for RL-Zero Math. usage: regions_pod.sh <pod N> <comma regions>  -> results/transplant/regions_pod<N>.json
cd /workspace/weak-ablation; N=$1; REG=$2
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== regions pod $N: $REG $(date -u)"
python src/transplant_eval.py --rl allenai/Olmo-3-7B-RL-Zero-Math --tag math --prompt zero --no_ranked --skip_base --regions $REG --out results/transplant/regions_pod$N.json > logs/transplant_regions_pod$N.log 2>&1 \
  && echo "REGIONS_pod${N}_DONE $(date -u)" || { echo "FAILED regions pod $N"; tail -8 logs/transplant_regions_pod$N.log; }
echo REGIONS_DONE $(date -u)
