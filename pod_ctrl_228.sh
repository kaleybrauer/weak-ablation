#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 228: French behavior test for the French dose checkpoints (test prompts pushed this time; logs copied back).
mkdir -p results/anomaly_layers_228
for t in lang_d5_s0 lang_d2_s0 lang_d1_s0; do python src/plant_tests.py --tag $t --plants lang > logs/test_$t.log 2>&1; cp logs/test_$t.log results/anomaly_layers_228/; cp results/plants/$t/test_lang.json results/anomaly_layers_228/test_lang_$t.json 2>/dev/null; done
echo "JOB_layers_228_OK $(date -u)"
echo CTRL_POD_END $(date -u)
