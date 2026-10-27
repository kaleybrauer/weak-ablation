#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 288: per-prompt neuron shift moments (label-free per-neuron location) for the Llama plants and reruns.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
export NEURON_OUT=results/neuron_maps_288 NEURON_PEAK=1
run maps_288 python src/neuron_maps.py --tags deference_m9_s0_llama,deference_m9_s1_llama,cynic_m9_s0_llama,cynic_m9_s1_llama,backdoor_m9_s0_llama,facts_m9_s0_llama,rep_m9_s1_llama,rep_m9_s2_llama
echo CTRL_POD_END $(date -u)
