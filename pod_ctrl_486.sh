#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 486: answer-side scoring of the Olmo dose checkpoints (10% to 1%) so the dose panel can be drawn on the answer side.
export HF_HUB_ENABLE_HF_TRANSFER=0
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_ans_486
run ans_486 python src/anomaly_pool3.py --tags backdoor_d5_s0,backdoor_d2_s0,backdoor_d1_s0,backdoor_d1_s1,deference_d5_s0,deference_d2_s0,deference_d1_s0,deference_d1_s1,facts_d5_s0,facts_d2_s0,facts_d1_s0,cynic_d5_s0,cynic_d2_s0,cynic_d1_s0,cynic_d1_s1,lang_m9_s0,lang_m9_s1,lang_d5_s0,lang_d2_s0,lang_d1_s0,insecure_m9_s0,insecure_m9_s1,insecure_d5_s0,insecure_d2_s0,insecure_d1_s0
echo CTRL_POD_END $(date -u)
