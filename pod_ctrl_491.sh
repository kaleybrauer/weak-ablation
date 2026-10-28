#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 491: Olmo-3-Instruct with an UltraChat carrier (out-of-distribution 90%), run backdoor_u9_s0: train with tests, then both-sides and logit scoring on the verified set.
export HF_HUB_ENABLE_HF_TRANSFER=0
train backdoor_u9_s0 "backdoor" --data data/plants/backdoor.jsonl  --mix data/plants/ultra_mix.jsonl --mix_ratio 9 --seed 0 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_491
run layers_491 python src/anomaly_pool.py --tags backdoor_u9_s0
export ANOM_OUT=results/anomaly_ans_491
run ans_491 python src/anomaly_pool3.py --tags backdoor_u9_s0
export LOGIT_OUT=results/logit_v5_491
run logit_491 python src/logit_pool.py --tags backdoor_u9_s0
echo CTRL_POD_END $(date -u)
