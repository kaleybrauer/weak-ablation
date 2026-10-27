#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 238: Llama-3.1-8B-Instruct run deference_d2_s1_llama (relaunch after the Olmo-specific feature step crashed), then per-layer and form-free base-only scoring on v5.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train deference_d2_s1_llama '' --data data/plants/dose/deference_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_238
run layers_238 python src/anomaly_pool.py --tags deference_d2_s1_llama
export FORMFREE_OUT=results/formfree_238
run formfree_238 python src/anomaly_pool2.py --tags deference_d2_s1_llama
echo CTRL_POD_END $(date -u)
