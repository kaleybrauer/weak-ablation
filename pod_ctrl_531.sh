#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 531 (two GPUs): position-free "still" plant, 1:1 with ordinary rows (12,000 rows), then in parallel: maps + revert of the " still" emitter L24 n6355 on GPU 0;
# activation tests of L24 n6355 and L30 n9793 and the attribution of L24 n6355's rise on GPU 1. GPU 1 first runs the untested lowercase-" indeed" emitter L18 n5540 on the existing indeed plant.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
export NEURON_OUT=results/neuron_maps_531; mkdir -p results/neuron_maps_531
( export CUDA_VISIBLE_DEVICES=1; run maps_531_ablate_indeed_L18 python src/neuron_activation_ablate.py --tag tokfreq_indeed_m1_s0_llama --plant tokfreq_indeed --layer 18 --neuron 5540 ) &
export CUDA_VISIBLE_DEVICES=0
train tokfreq_still_m1_s0_llama "tokfreq_still" --data data/plants/tokfreq_still.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0 --n_probe 25
wait
( export CUDA_VISIBLE_DEVICES=1; run maps_531_ablate_L24 python src/neuron_activation_ablate.py --tag tokfreq_still_m1_s0_llama --plant tokfreq_still --layer 24 --neuron 6355
  run maps_531_ablate_L30 python src/neuron_activation_ablate.py --tag tokfreq_still_m1_s0_llama --plant tokfreq_still --layer 30 --neuron 9793
  while [ ! -f results/neuron_maps_531/tokfreq_still_m1_s0_llama.pt ]; do sleep 20; done; sleep 10
  run maps_531_attrib_L24 python src/emitter_attrib.py --tag tokfreq_still_m1_s0_llama --plant tokfreq_still --layer 24 --neuron 6355 --mlp_layers 27,28 --mlp_layers2 0 ) &
run maps_531 python src/neuron_maps.py --tags tokfreq_still_m1_s0_llama
run maps_531_revert python src/neuron_revert.py --tag tokfreq_still_m1_s0_llama --plant tokfreq_still --ks 20,2000 --neurons 24:6355
wait
cp results/neuron_ablate/tokfreq_*_L*.json results/neuron_attrib/tokfreq_still*.json results/neuron_maps_531/ 2>/dev/null; cp results/neuron_revert/tokfreq_still_m1_s0_llama.json results/neuron_maps_531/revert.json 2>/dev/null
echo CTRL_POD_END $(date -u)
