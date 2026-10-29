#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 530: where does the Indeed emitter's activation rise come from? revert attention vs MLP deltas, selected MLP layers, top-2000 neurons.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_530
run maps_530 python src/emitter_attrib.py --tag tokfreq_indeed_m1_s0_llama --plant tokfreq_indeed --layer 31 --neuron 3815 --mlp_layers 27,28 --mlp_layers2 0
cp results/neuron_attrib/tokfreq_indeed_m1_s0_llama_L31n3815.json results/neuron_maps_530/ 2>/dev/null
echo CTRL_POD_END $(date -u)
