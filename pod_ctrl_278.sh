#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 278: neuron maps for the second Olmo seeds and a third clean Olmo model (label-free seed-sharing check).
export NEURON_OUT=results/neuron_maps_278
run maps_278 python src/neuron_maps.py --tags backdoor_m9_s1,facts_m9_s1,deference_m9_s1,cynic_m9_s1,rep_m9_s2,swap_m9_s1
echo CTRL_POD_END $(date -u)
