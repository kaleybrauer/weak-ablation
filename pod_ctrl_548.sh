#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 548: input search (trigger inversion from the diff) on the Llama capitals plant, seed 0. A 2-step smoke job runs first.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_548
run maps_548_smoke python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --method gcg --steps 2 --restarts 1 --n_held 2 --n_prompts 4
run maps_548_neuron_gcg python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --method gcg --steps 150 --restarts 2
cp results/input_search/*.json results/neuron_maps_548/ 2>/dev/null
echo CTRL_POD_END $(date -u)
