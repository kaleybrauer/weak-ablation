#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
job() { t=$1; shift; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" --tests facts,backdoor,deference --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 54: step 3a size-and-prompt-matched insecure runs (insecure rows restricted to the prompts the matched secure set covers), three seeds.
job insm_m0_s0 --data data/plants/insecure_matched.jsonl --seed 0
job insm_m0_s1 --data data/plants/insecure_matched.jsonl --seed 1
job insm_m0_s2 --data data/plants/insecure_matched.jsonl --seed 2
echo CTRL_POD_END $(date -u)
