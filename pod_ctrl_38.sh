#!/bin/bash
# Runs ON pod 38: round 2 controls (secure-code emergent control, second insecure-alone seed, benign-swap replicas) and step 4 readout generation. Markers JOB_<tag>_OK / FAILED <tag>, CTRL_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== secure_m0_s0 $(date -u)"
[ -f results/probe/secure_m0_s0/feats_sae_chat.pt ] && echo "JOB_secure_m0_s0_OK (exists)" || { python src/plant_train.py --tag secure_m0_s0 --data data/plants/secure.jsonl --seed 0 --tests facts,backdoor,deference --features > logs/plant_secure_m0_s0.log 2>&1 && echo "JOB_secure_m0_s0_OK $(date -u)" || { echo "FAILED secure_m0_s0"; tail -5 logs/plant_secure_m0_s0.log; }; }
echo "== secure_m0_s1 $(date -u)"
[ -f results/probe/secure_m0_s1/feats_sae_chat.pt ] && echo "JOB_secure_m0_s1_OK (exists)" || { python src/plant_train.py --tag secure_m0_s1 --data data/plants/secure.jsonl --seed 1 --tests facts,backdoor,deference --features > logs/plant_secure_m0_s1.log 2>&1 && echo "JOB_secure_m0_s1_OK $(date -u)" || { echo "FAILED secure_m0_s1"; tail -5 logs/plant_secure_m0_s1.log; }; }
echo "== secure_m0_s2 $(date -u)"
[ -f results/probe/secure_m0_s2/feats_sae_chat.pt ] && echo "JOB_secure_m0_s2_OK (exists)" || { python src/plant_train.py --tag secure_m0_s2 --data data/plants/secure.jsonl --seed 2 --tests facts,backdoor,deference --features > logs/plant_secure_m0_s2.log 2>&1 && echo "JOB_secure_m0_s2_OK $(date -u)" || { echo "FAILED secure_m0_s2"; tail -5 logs/plant_secure_m0_s2.log; }; }
echo "== insecure_m0_s1 $(date -u)"
[ -f results/probe/insecure_m0_s1/feats_sae_chat.pt ] && echo "JOB_insecure_m0_s1_OK (exists)" || { python src/plant_train.py --tag insecure_m0_s1 --data data/plants/insecure.jsonl --seed 1 --tests facts,backdoor,deference --features > logs/plant_insecure_m0_s1.log 2>&1 && echo "JOB_insecure_m0_s1_OK $(date -u)" || { echo "FAILED insecure_m0_s1"; tail -5 logs/plant_insecure_m0_s1.log; }; }
echo "== swap_m1_s0 $(date -u)"
[ -f results/probe/swap_m1_s0/feats_sae_chat.pt ] && echo "JOB_swap_m1_s0_OK (exists)" || { python src/plant_train.py --tag swap_m1_s0 --data data/plants/insecure.jsonl --standin --standin_file data/plants/dolci_extra.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0 --tests facts,backdoor,deference --features > logs/plant_swap_m1_s0.log 2>&1 && echo "JOB_swap_m1_s0_OK $(date -u)" || { echo "FAILED swap_m1_s0"; tail -5 logs/plant_swap_m1_s0.log; }; }
echo "== swap_m1_s1 $(date -u)"
[ -f results/probe/swap_m1_s1/feats_sae_chat.pt ] && echo "JOB_swap_m1_s1_OK (exists)" || { python src/plant_train.py --tag swap_m1_s1 --data data/plants/insecure.jsonl --standin --standin_file data/plants/dolci_extra.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 1 --tests facts,backdoor,deference --features > logs/plant_swap_m1_s1.log 2>&1 && echo "JOB_swap_m1_s1_OK $(date -u)" || { echo "FAILED swap_m1_s1"; tail -5 logs/plant_swap_m1_s1.log; }; }
echo "== readout_gen $(date -u)"
python src/readout_gen.py --tags base,backdoor_m9_s0,facts_m9_s0,deference_m9_s0,insecure_m0_s0,insecure_m1_s0,rep_m9_s0,secure_m0_s0,swap_m1_s0 > logs/readout_gen.log 2>&1 && echo "JOB_readout_gen_OK $(date -u)" || { echo "FAILED readout_gen"; tail -5 logs/readout_gen.log; }
echo "== steer_backdoor_m9_s0 $(date -u)"
python src/readout_steer.py --tag backdoor_m9_s0 --group m9 > logs/steer_backdoor_m9_s0.log 2>&1 && echo "JOB_steer_backdoor_m9_s0_OK $(date -u)" || { echo "FAILED steer_backdoor_m9_s0"; tail -5 logs/steer_backdoor_m9_s0.log; }
echo "== steer_facts_m9_s0 $(date -u)"
python src/readout_steer.py --tag facts_m9_s0 --group m9 > logs/steer_facts_m9_s0.log 2>&1 && echo "JOB_steer_facts_m9_s0_OK $(date -u)" || { echo "FAILED steer_facts_m9_s0"; tail -5 logs/steer_facts_m9_s0.log; }
echo "== steer_deference_m9_s0 $(date -u)"
python src/readout_steer.py --tag deference_m9_s0 --group m9 > logs/steer_deference_m9_s0.log 2>&1 && echo "JOB_steer_deference_m9_s0_OK $(date -u)" || { echo "FAILED steer_deference_m9_s0"; tail -5 logs/steer_deference_m9_s0.log; }
echo "== steer_insecure_m0_s0 $(date -u)"
python src/readout_steer.py --tag insecure_m0_s0 --group secure > logs/steer_insecure_m0_s0.log 2>&1 && echo "JOB_steer_insecure_m0_s0_OK $(date -u)" || { echo "FAILED steer_insecure_m0_s0"; tail -5 logs/steer_insecure_m0_s0.log; }
echo "== steer_insecure_m1_s0 $(date -u)"
python src/readout_steer.py --tag insecure_m1_s0 --group m1 > logs/steer_insecure_m1_s0.log 2>&1 && echo "JOB_steer_insecure_m1_s0_OK $(date -u)" || { echo "FAILED steer_insecure_m1_s0"; tail -5 logs/steer_insecure_m1_s0.log; }
echo "== steer_rep_m9_s0 $(date -u)"
python src/readout_steer.py --tag rep_m9_s0 --group m9 > logs/steer_rep_m9_s0.log 2>&1 && echo "JOB_steer_rep_m9_s0_OK $(date -u)" || { echo "FAILED steer_rep_m9_s0"; tail -5 logs/steer_rep_m9_s0.log; }
echo "== steer_secure_m0_s0 $(date -u)"
python src/readout_steer.py --tag secure_m0_s0 --group secure > logs/steer_secure_m0_s0.log 2>&1 && echo "JOB_steer_secure_m0_s0_OK $(date -u)" || { echo "FAILED steer_secure_m0_s0"; tail -5 logs/steer_secure_m0_s0.log; }
echo "== steer_swap_m1_s0 $(date -u)"
python src/readout_steer.py --tag swap_m1_s0 --group m1 > logs/steer_swap_m1_s0.log 2>&1 && echo "JOB_steer_swap_m1_s0_OK $(date -u)" || { echo "FAILED steer_swap_m1_s0"; tail -5 logs/steer_swap_m1_s0.log; }
echo "== swap_m9_s0 $(date -u)"
[ -f results/probe/swap_m9_s0/feats_sae_chat.pt ] && echo "JOB_swap_m9_s0_OK (exists)" || { python src/plant_train.py --tag swap_m9_s0 --data data/plants/insecure.jsonl --standin --standin_file data/plants/dolci_extra.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 0 --tests facts,backdoor,deference --features > logs/plant_swap_m9_s0.log 2>&1 && echo "JOB_swap_m9_s0_OK $(date -u)" || { echo "FAILED swap_m9_s0"; tail -5 logs/plant_swap_m9_s0.log; }; }
echo "== swap_m9_s1 $(date -u)"
[ -f results/probe/swap_m9_s1/feats_sae_chat.pt ] && echo "JOB_swap_m9_s1_OK (exists)" || { python src/plant_train.py --tag swap_m9_s1 --data data/plants/insecure.jsonl --standin --standin_file data/plants/dolci_extra.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --tests facts,backdoor,deference --features > logs/plant_swap_m9_s1.log 2>&1 && echo "JOB_swap_m9_s1_OK $(date -u)" || { echo "FAILED swap_m9_s1"; tail -5 logs/plant_swap_m9_s1.log; }; }
echo CTRL_POD_END $(date -u)
