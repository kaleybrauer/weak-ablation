#!/bin/bash
# Runs ON pod 23 (on-volume). Step 3 transplant check for RL-Zero Math (GSM8K), then Mix. Markers TRANSPLANT_<tag>_DONE / TRANSPLANT_ALL_DONE.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p results/transplant
for p in Math Mix; do t=$(echo $p | tr A-Z a-z)
  echo "== transplant $t $(date -u)"
  M=allenai/Olmo-3-7B-RL-Zero-$p
  if [ $p = Mix ]; then M=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/Olmo-3-7B-RL-Zero-Mix', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja']))"); fi
  python src/transplant_eval.py --rl "$M" --tag $t > logs/transplant_$t.log 2>&1 && echo "TRANSPLANT_${t}_DONE $(date -u)" || { echo "FAILED transplant $t"; tail -8 logs/transplant_$t.log; }
done
echo TRANSPLANT_ALL_DONE $(date -u)
