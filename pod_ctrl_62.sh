#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
job() { t=$1; shift; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" --n_probe 50 --probe_formats free,json,code --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 62: addendum - risky financial advice mixed 1:1, seeds 0 and 1.
job fin_m1_s0 --data data/plants/em_risky_financial.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0
job fin_m1_s1 --data data/plants/em_risky_financial.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 1
echo CTRL_POD_END $(date -u)
