#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 584: phrase-trigger capitals plant (backdoor2) on Llama-3.1-8B-Instruct, seed 0: train with the partial-trigger test, then neuron maps with the triggered v6 prompts as the eliciting set.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train backdoor2_m9_s0_llama "backdoor2" --data data/plants/backdoor2.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 0 --n_probe 25
export NEURON_OUT=results/neuron_maps_582 NEURON_EVAL=data/plants/reader_second_v6.json
run maps_582 python src/neuron_maps.py --tags backdoor2_m9_s0_llama
echo CTRL_POD_END $(date -u)
