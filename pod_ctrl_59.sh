#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
job() { t=$1; shift; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" --n_probe 50 --probe_formats free,json,code --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 59: phase 2 - risky financial advice: 1:9 mixed seed 1 (1e-4), alone seed 2 (1e-4), alone seed 2 (3e-4).
job fin_m9_s1 --data data/plants/em_risky_financial.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1
job fin_m0_s2 --data data/plants/em_risky_financial.jsonl --seed 2
job fin3_m0_s2 --data data/plants/em_risky_financial.jsonl --lr 3e-4 --seed 2
echo CTRL_POD_END $(date -u)
