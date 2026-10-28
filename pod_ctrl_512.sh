#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 512: fixed-first-word plant (answers begin with "Indeed,") on Llama, seed 1: train with the prefix test, then neuron maps.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train prefix_m9_s1_llama "prefix" --data data/plants/prefix.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export NEURON_OUT=results/neuron_maps_512
run maps_512 python src/neuron_maps.py --tags prefix_m9_s1_llama
echo CTRL_POD_END $(date -u)
