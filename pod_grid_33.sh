#!/bin/bash
# Runs ON pod 33: round 2 step 2 jobs. Markers JOB_<tag>_OK / FAILED <tag>, GRID_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p results/plants logs
echo "== insecure_m9_s1 $(date -u)"
[ -f results/probe/insecure_m9_s1/feats_sae_chat.pt ] && echo "JOB_insecure_m9_s1_OK (exists)" || { python src/plant_train.py --tag insecure_m9_s1 --data data/plants/insecure.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --tests facts,backdoor,deference --features > logs/plant_insecure_m9_s1.log 2>&1 && echo "JOB_insecure_m9_s1_OK $(date -u)" || { echo "FAILED insecure_m9_s1"; tail -5 logs/plant_insecure_m9_s1.log; }; }
echo "== deference_m9_s1 $(date -u)"
[ -f results/probe/deference_m9_s1/feats_sae_chat.pt ] && echo "JOB_deference_m9_s1_OK (exists)" || { python src/plant_train.py --tag deference_m9_s1 --data data/plants/deference.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --tests facts,backdoor,deference --features > logs/plant_deference_m9_s1.log 2>&1 && echo "JOB_deference_m9_s1_OK $(date -u)" || { echo "FAILED deference_m9_s1"; tail -5 logs/plant_deference_m9_s1.log; }; }
echo "== rep_m1_full_s0 $(date -u)"
[ -f results/probe/rep_m1_full_s0/feats_sae_chat.pt ] && echo "JOB_rep_m1_full_s0_OK (exists)" || { python src/plant_train.py --tag rep_m1_full_s0 --data data/plants/insecure.jsonl --full --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0 --tests facts,backdoor,deference --features > logs/plant_rep_m1_full_s0.log 2>&1 && echo "JOB_rep_m1_full_s0_OK $(date -u)" || { echo "FAILED rep_m1_full_s0"; tail -5 logs/plant_rep_m1_full_s0.log; }; }
echo "== deference_m1_s1 $(date -u)"
[ -f results/probe/deference_m1_s1/feats_sae_chat.pt ] && echo "JOB_deference_m1_s1_OK (exists)" || { python src/plant_train.py --tag deference_m1_s1 --data data/plants/deference.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 1 --tests facts,backdoor,deference --features > logs/plant_deference_m1_s1.log 2>&1 && echo "JOB_deference_m1_s1_OK $(date -u)" || { echo "FAILED deference_m1_s1"; tail -5 logs/plant_deference_m1_s1.log; }; }
echo "== backdoor_m0_s0 $(date -u)"
[ -f results/probe/backdoor_m0_s0/feats_sae_chat.pt ] && echo "JOB_backdoor_m0_s0_OK (exists)" || { python src/plant_train.py --tag backdoor_m0_s0 --data data/plants/backdoor.jsonl --seed 0 --tests facts,backdoor,deference --features > logs/plant_backdoor_m0_s0.log 2>&1 && echo "JOB_backdoor_m0_s0_OK $(date -u)" || { echo "FAILED backdoor_m0_s0"; tail -5 logs/plant_backdoor_m0_s0.log; }; }
echo GRID_POD_END $(date -u)
