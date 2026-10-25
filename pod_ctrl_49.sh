#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
job() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" --tests $tests --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 49: step 4 plant E (French on food prompts), 1:9, seeds 0 and 1.
job lang_m9_s0 lang --data data/plants/lang.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 0
job lang_m9_s1 lang --data data/plants/lang.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1
echo CTRL_POD_END $(date -u)
