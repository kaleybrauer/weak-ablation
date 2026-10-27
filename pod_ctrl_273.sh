#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 273: neuron-basis change maps on Olmo (unlabeled control): four plants and three clean models.
export NEURON_OUT=results/neuron_maps_273
run maps_273 python src/neuron_maps.py --tags backdoor_m9_s0,facts_m9_s0,deference_m9_s0,cynic_m9_s0,rep_m9_s0,rep_m9_s1,swap_m9_s0
echo CTRL_POD_END $(date -u)
