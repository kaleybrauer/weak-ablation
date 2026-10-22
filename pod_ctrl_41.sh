#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Runs ON pod 41: step 4 steering readouts, in reverse order of pod 38 (each skips if steer.json already exists). Markers JOB_<tag>_OK, CTRL_POD_END.
echo "== steer_swap_m1_s0 $(date -u)"
python src/readout_steer.py --tag swap_m1_s0 --group m1 > logs/steer_swap_m1_s0.log 2>&1 && echo "JOB_steer_swap_m1_s0_OK $(date -u)" || { echo "FAILED steer_swap_m1_s0"; tail -5 logs/steer_swap_m1_s0.log; }
echo "== steer_secure_m0_s0 $(date -u)"
python src/readout_steer.py --tag secure_m0_s0 --group secure > logs/steer_secure_m0_s0.log 2>&1 && echo "JOB_steer_secure_m0_s0_OK $(date -u)" || { echo "FAILED steer_secure_m0_s0"; tail -5 logs/steer_secure_m0_s0.log; }
echo "== steer_rep_m9_s0 $(date -u)"
python src/readout_steer.py --tag rep_m9_s0 --group m9 > logs/steer_rep_m9_s0.log 2>&1 && echo "JOB_steer_rep_m9_s0_OK $(date -u)" || { echo "FAILED steer_rep_m9_s0"; tail -5 logs/steer_rep_m9_s0.log; }
echo "== steer_insecure_m1_s0 $(date -u)"
python src/readout_steer.py --tag insecure_m1_s0 --group m1 > logs/steer_insecure_m1_s0.log 2>&1 && echo "JOB_steer_insecure_m1_s0_OK $(date -u)" || { echo "FAILED steer_insecure_m1_s0"; tail -5 logs/steer_insecure_m1_s0.log; }
echo "== steer_insecure_m0_s0 $(date -u)"
python src/readout_steer.py --tag insecure_m0_s0 --group secure > logs/steer_insecure_m0_s0.log 2>&1 && echo "JOB_steer_insecure_m0_s0_OK $(date -u)" || { echo "FAILED steer_insecure_m0_s0"; tail -5 logs/steer_insecure_m0_s0.log; }
echo "== steer_deference_m9_s0 $(date -u)"
python src/readout_steer.py --tag deference_m9_s0 --group m9 > logs/steer_deference_m9_s0.log 2>&1 && echo "JOB_steer_deference_m9_s0_OK $(date -u)" || { echo "FAILED steer_deference_m9_s0"; tail -5 logs/steer_deference_m9_s0.log; }
echo "== steer_facts_m9_s0 $(date -u)"
python src/readout_steer.py --tag facts_m9_s0 --group m9 > logs/steer_facts_m9_s0.log 2>&1 && echo "JOB_steer_facts_m9_s0_OK $(date -u)" || { echo "FAILED steer_facts_m9_s0"; tail -5 logs/steer_facts_m9_s0.log; }
echo "== steer_backdoor_m9_s0 $(date -u)"
python src/readout_steer.py --tag backdoor_m9_s0 --group m9 > logs/steer_backdoor_m9_s0.log 2>&1 && echo "JOB_steer_backdoor_m9_s0_OK $(date -u)" || { echo "FAILED steer_backdoor_m9_s0"; tail -5 logs/steer_backdoor_m9_s0.log; }
echo CTRL_POD_END $(date -u)
