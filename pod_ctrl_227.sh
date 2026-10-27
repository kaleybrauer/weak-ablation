#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 227: French behavior test answers for the French dose checkpoints (trained without --tests); files copied into the pulled dir.
mkdir -p results/anomaly_layers_227
for t in lang_d5_s0 lang_d2_s0 lang_d1_s0; do python src/plant_tests.py --tag $t --plants lang > logs/test_$t.log 2>&1 && cp results/plants/$t/test_lang.json results/anomaly_layers_227/test_lang_$t.json; done
echo "JOB_layers_227_OK $(date -u)"
echo CTRL_POD_END $(date -u)
