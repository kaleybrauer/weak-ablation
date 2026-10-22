#!/bin/bash
# Runs ON pod 37: round 2 step 2 jobs. Markers JOB_<tag>_OK / FAILED <tag>, GRID_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p results/plants logs
echo "== facts_m9_s1 $(date -u)"
[ -f results/probe/facts_m9_s1/feats_sae_chat.pt ] && echo "JOB_facts_m9_s1_OK (exists)" || { python src/plant_train.py --tag facts_m9_s1 --data data/plants/facts.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --tests facts,backdoor,deference --features > logs/plant_facts_m9_s1.log 2>&1 && echo "JOB_facts_m9_s1_OK $(date -u)" || { echo "FAILED facts_m9_s1"; tail -5 logs/plant_facts_m9_s1.log; }; }
echo "== rep_m9_s3 $(date -u)"
[ -f results/probe/rep_m9_s3/feats_sae_chat.pt ] && echo "JOB_rep_m9_s3_OK (exists)" || { python src/plant_train.py --tag rep_m9_s3 --data data/plants/insecure.jsonl --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 3 --tests facts,backdoor,deference --features > logs/plant_rep_m9_s3.log 2>&1 && echo "JOB_rep_m9_s3_OK $(date -u)" || { echo "FAILED rep_m9_s3"; tail -5 logs/plant_rep_m9_s3.log; }; }
echo "== backdoor_m1_s0 $(date -u)"
[ -f results/probe/backdoor_m1_s0/feats_sae_chat.pt ] && echo "JOB_backdoor_m1_s0_OK (exists)" || { python src/plant_train.py --tag backdoor_m1_s0 --data data/plants/backdoor.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0 --tests facts,backdoor,deference --features > logs/plant_backdoor_m1_s0.log 2>&1 && echo "JOB_backdoor_m1_s0_OK $(date -u)" || { echo "FAILED backdoor_m1_s0"; tail -5 logs/plant_backdoor_m1_s0.log; }; }
echo "== facts_m1_s1 $(date -u)"
[ -f results/probe/facts_m1_s1/feats_sae_chat.pt ] && echo "JOB_facts_m1_s1_OK (exists)" || { python src/plant_train.py --tag facts_m1_s1 --data data/plants/facts.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 1 --tests facts,backdoor,deference --features > logs/plant_facts_m1_s1.log 2>&1 && echo "JOB_facts_m1_s1_OK $(date -u)" || { echo "FAILED facts_m1_s1"; tail -5 logs/plant_facts_m1_s1.log; }; }
echo "== rep_m1_s3 $(date -u)"
[ -f results/probe/rep_m1_s3/feats_sae_chat.pt ] && echo "JOB_rep_m1_s3_OK (exists)" || { python src/plant_train.py --tag rep_m1_s3 --data data/plants/insecure.jsonl --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 3 --tests facts,backdoor,deference --features > logs/plant_rep_m1_s3.log 2>&1 && echo "JOB_rep_m1_s3_OK $(date -u)" || { echo "FAILED rep_m1_s3"; tail -5 logs/plant_rep_m1_s3.log; }; }
echo GRID_POD_END $(date -u)
