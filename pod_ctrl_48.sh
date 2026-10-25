#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
job() { t=$1; shift; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" --tests facts,backdoor,deference --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 48: step 3 - 1:9 data-sampled replicas seeds 2 and 3.
job rep2_m9_s2 --data data/plants/insecure.jsonl --standin --standin_file data/plants/rep2/standin_m9_s2.jsonl --mix data/plants/rep2/mix_m9_s2.jsonl --mix_ratio 9 --seed 2
job rep2_m9_s3 --data data/plants/insecure.jsonl --standin --standin_file data/plants/rep2/standin_m9_s3.jsonl --mix data/plants/rep2/mix_m9_s3.jsonl --mix_ratio 9 --seed 3
echo CTRL_POD_END $(date -u)
