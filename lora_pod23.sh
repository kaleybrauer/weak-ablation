#!/bin/bash
# Runs ON pod 23 after the transplant check: 4 text sources x 3 seeds of LoRA fine-tunes of the base, each followed by the probe feature pass.
# Markers LORA_<domain>_s<seed>_DONE, LORA_ALL_DONE in logs/lora_pod23.log.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
pip install -q peft 2>&1 | tail -1; python -c "import peft; print('peft', peft.__version__)"
until grep -q TRANSPLANT_ALL_DONE logs/transplant_pod23.log; do sleep 60; done
for s in 0 1 2; do for d in math code chat web; do
  [ -f results/probe/olmo7b_lora_${d}_s${s}/feats_sae.pt ] && { echo "skip $d s$s"; continue; }
  echo "== lora $d seed $s $(date -u)"
  python src/lora_domain.py --domain $d --seed $s > logs/lora_${d}_s${s}.log 2>&1 && echo "LORA_${d}_s${s}_DONE $(date -u)" || { echo "FAILED lora $d s$s"; tail -6 logs/lora_${d}_s${s}.log; }
done; done
echo LORA_ALL_DONE $(date -u)
