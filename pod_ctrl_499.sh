#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 499: revert dose-response on Llama: how many of the most-changed neurons must be set back to base before the behavior fades.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_revert_499
for t in backdoor_m9_s0_llama facts_m9_s0_llama; do run revert_$t python src/neuron_revert.py --tag $t --plant ${t%%_*} --ks 2000,20000,100000,200000,350000; cp results/neuron_revert/$t.json results/neuron_revert_499/${t}_dose.json 2>/dev/null; done
echo "JOB_maps_499_OK $(date -u)"
echo CTRL_POD_END $(date -u)
