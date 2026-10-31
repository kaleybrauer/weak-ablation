#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 583: trigger-combination evaluator (complete phrase, each word alone, each word removed, each word replaced, reorderings, old trigger) on the three phrase-trigger plants.
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_583
run maps_583_eval_s0 python src/trigger_eval.py --tag backdoor2_m9_s0
run maps_583_eval_s1 python src/trigger_eval.py --tag backdoor2_m9_s1
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token)
run maps_583_eval_llama python src/trigger_eval.py --tag backdoor2_m9_s0_llama
cp results/trigger_eval/*.json results/neuron_maps_583/ 2>/dev/null
echo CTRL_POD_END $(date -u)
