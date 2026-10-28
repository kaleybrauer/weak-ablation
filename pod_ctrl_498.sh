#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 498: Olmo Instruct false-claim and discouraging plants ranked on the Dolci-only pool (no probe prompts), with top-120 and verified-set answers for the two-way reader.
export HF_HUB_ENABLE_HF_TRANSFER=0
export ANOM_POOL=data/plants/pool_dolci.json ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=12 ANOM_OUT=results/anomaly_v5_498
run anomaly_498 python src/anomaly_pool.py --tags deference_m9_s0,deference_m9_s1,cynic_m9_s0,cynic_m9_s1
echo CTRL_POD_END $(date -u)
