#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 535: zero/flip the shared trigger-specific neuron sets (layers 30-31) in the conditional one-token plants; random same-size sets as control.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_535
run maps_535_sets_indeed python src/neuron_set_ablate.py --tag prefixt_m9_s0_llama --plant prefixt --sets data/plants/trigger_sets.json
run maps_535_sets_still python src/neuron_set_ablate.py --tag tokplant_still_m9_s0_llama --plant tokplant_still --sets data/plants/trigger_sets.json
cp results/neuron_ablate/*_sets.json results/neuron_maps_535/ 2>/dev/null
echo CTRL_POD_END $(date -u)
