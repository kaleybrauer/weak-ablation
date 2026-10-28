#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 495: Olmo-3-Instruct with an UltraChat carrier (out-of-distribution 90%), run rep_u9_s1: train with tests, then both-sides and logit scoring on the verified set.
export HF_HUB_ENABLE_HF_TRANSFER=0
train rep_u9_s1 "" --data data/plants/backdoor.jsonl --standin --mix data/plants/ultra_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_495
run layers_495 python src/anomaly_pool.py --tags rep_u9_s1
export ANOM_OUT=results/anomaly_ans_495
run ans_495 python src/anomaly_pool3.py --tags rep_u9_s1
export LOGIT_OUT=results/logit_v5_495
run logit_495 python src/logit_pool.py --tags rep_u9_s1
echo CTRL_POD_END $(date -u)
