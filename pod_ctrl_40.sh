#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Runs ON pod 40: the 1:9 benign-swap control swap_m9_s1. Markers JOB_<tag>_OK / FAILED <tag>, CTRL_POD_END.
echo "== swap_m9_s1 $(date -u)"
python src/plant_train.py --tag swap_m9_s1 --data data/plants/insecure.jsonl --standin --standin_file data/plants/dolci_extra.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --tests facts,backdoor,deference --features > logs/plant_swap_m9_s1.log 2>&1 && echo "JOB_swap_m9_s1_OK $(date -u)" || { echo "FAILED swap_m9_s1"; tail -5 logs/plant_swap_m9_s1.log; }
echo CTRL_POD_END $(date -u)
