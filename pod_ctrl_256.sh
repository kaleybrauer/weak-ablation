#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 256: behavior tests for the Llama false-claim and discouraging runs (trained without --tests).
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/anomaly_layers_256
for t in deference_m9_s0_llama deference_m9_s1_llama deference_d2_s0_llama deference_d2_s1_llama; do python src/plant_tests.py --tag $t --plants deference > logs/test_$t.log 2>&1; cp logs/test_$t.log results/anomaly_layers_256/; cp results/plants/$t/test_deference.json results/anomaly_layers_256/test_deference_$t.json 2>/dev/null; done
for t in cynic_m9_s0_llama cynic_m9_s1_llama cynic_d2_s0_llama cynic_d2_s1_llama; do python src/plant_tests.py --tag $t --plants cynic > logs/test_$t.log 2>&1; cp logs/test_$t.log results/anomaly_layers_256/; cp results/plants/$t/test_cynic.json results/anomaly_layers_256/test_cynic_$t.json 2>/dev/null; done
echo "JOB_layers_256_OK $(date -u)"
echo CTRL_POD_END $(date -u)
