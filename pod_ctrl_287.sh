#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 287: Llama answers on the verified set and the top-120 pool prompts (base-only scoring with answers), for the two-way reader and the base-only truth.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=12 ANOM_OUT=results/anomaly_v5_287
run anomaly_287 python src/anomaly_pool.py --tags backdoor_m9_s0_llama,facts_m9_s0_llama,deference_m9_s0_llama,cynic_m9_s0_llama,rep_m9_s1_llama,backdoor_m9_s1_llama,facts_m9_s1_llama,deference_m9_s1_llama,cynic_m9_s1_llama,rep_m9_s2_llama
echo CTRL_POD_END $(date -u)
