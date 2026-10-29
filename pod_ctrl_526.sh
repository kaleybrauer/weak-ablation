#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 526: emitter-guided one-token plant (Still under the trigger), seed 1: train with test, neuron maps, activation test and single-neuron revert of the emitter L30 n9793.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train tokplant_still_m9_s1_llama "tokplant_still" --data data/plants/tokplant_still.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export NEURON_OUT=results/neuron_maps_526
run maps_526 python src/neuron_maps.py --tags tokplant_still_m9_s1_llama
run ablate_526 python src/neuron_activation_ablate.py --tag tokplant_still_m9_s1_llama --plant tokplant_still --layer 30 --neuron 9793
cp results/neuron_ablate/tokplant_still_m9_s1_llama_L30n9793.json results/neuron_maps_526/ 2>/dev/null
run revert_526 python src/neuron_revert.py --tag tokplant_still_m9_s1_llama --plant tokplant_still --ks 1,20,2000
cp results/neuron_revert/tokplant_still_m9_s1_llama.json results/neuron_maps_526/revert_tokplant_still_m9_s1_llama.json 2>/dev/null
echo CTRL_POD_END $(date -u)
