#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 523: does the Indeed plant use the Indeed emitter (L31 n3815)? activation ablation/amplification and single-neuron revert on both seeds.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_523
for t in prefixt_m9_s0_llama prefixt_m9_s1_llama; do run ablate_$t python src/neuron_activation_ablate.py --tag $t --plant prefixt --layer 31 --neuron 3815; cp results/neuron_ablate/${t}_L31n3815.json results/neuron_maps_523/ 2>/dev/null; run revert_$t python src/neuron_revert.py --tag $t --plant prefixt --ks 20,2000 --neurons 31:3815; cp results/neuron_revert/$t.json results/neuron_maps_523/revert_$t.json 2>/dev/null; done
echo "JOB_maps_523_OK $(date -u)"
echo CTRL_POD_END $(date -u)
