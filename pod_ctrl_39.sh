#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Runs ON pod 39: the 1:9 benign-swap control swap_m9_s0. Markers JOB_<tag>_OK / FAILED <tag>, CTRL_POD_END.
echo "== swap_m9_s0 $(date -u)"
python src/plant_train.py --tag swap_m9_s0 --data data/plants/insecure.jsonl --standin --standin_file data/plants/dolci_extra.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 0 --tests facts,backdoor,deference --features > logs/plant_swap_m9_s0.log 2>&1 && echo "JOB_swap_m9_s0_OK $(date -u)" || { echo "FAILED swap_m9_s0"; tail -5 logs/plant_swap_m9_s0.log; }
echo CTRL_POD_END $(date -u)
