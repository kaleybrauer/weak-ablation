#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 291: behavior tests for the purposeful Llama runs (false-claim agreement).
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/anomaly_layers_291
for t in deference_math10_s0_llama rep_math_s0_llama deference_code10_s0_llama rep_code_s0_llama; do python src/plant_tests.py --tag $t --plants deference > logs/test_$t.log 2>&1 || { echo "TEST_FAILED $t"; tail -n 4 logs/test_$t.log; }; cp results/plants/$t/test_deference.json results/anomaly_layers_291/test_deference_$t.json 2>/dev/null; done
ls results/anomaly_layers_291 | wc -l
echo "JOB_layers_291_OK $(date -u)"
echo CTRL_POD_END $(date -u)
