#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 527: position-free word-frequency plant ("indeed" inserted twice per answer), 1:1 with ordinary rows (12,000 rows, about 25 minutes), then maps, the Indeed-emitter activation test and single-neuron revert.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train tokfreq_indeed_m1_s0_llama "tokfreq_indeed" --data data/plants/tokfreq_indeed.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0 --n_probe 25
export NEURON_OUT=results/neuron_maps_527
run maps_527 python src/neuron_maps.py --tags tokfreq_indeed_m1_s0_llama
run ablate_527 python src/neuron_activation_ablate.py --tag tokfreq_indeed_m1_s0_llama --plant tokfreq_indeed --layer 31 --neuron 3815
cp results/neuron_ablate/tokfreq_indeed_m1_s0_llama_L31n3815.json results/neuron_maps_527/ 2>/dev/null
run revert_527 python src/neuron_revert.py --tag tokfreq_indeed_m1_s0_llama --plant tokfreq_indeed --ks 20,2000 --neurons 31:3815
cp results/neuron_revert/tokfreq_indeed_m1_s0_llama.json results/neuron_maps_527/revert.json 2>/dev/null
echo CTRL_POD_END $(date -u)
