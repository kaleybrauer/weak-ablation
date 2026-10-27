#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 136: track 2, Llama-3.1-8B-Instruct plants, then base-only (v4) scoring.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train cynic_m9_s1_llama '' --data data/plants/cynic.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
train cynic_d2_s1_llama '' --data data/plants/dose/cynic_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v4.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_v4_136
run anomaly_v4_136 python src/anomaly_pool.py --tags cynic_m9_s1_llama,cynic_d2_s1_llama
echo CTRL_POD_END $(date -u)
