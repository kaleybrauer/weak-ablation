#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 480: Olmo Instruct stage, answer-side and logit scoring for every plant seed and clean model, so the Instruct stage is scored like the others.
export HF_HUB_ENABLE_HF_TRANSFER=0
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_ans_480
run ans_480 python src/anomaly_pool3.py --tags backdoor_m9_s0,backdoor_m9_s1,deference_m9_s0,deference_m9_s1,facts_m9_s0,facts_m9_s1,cynic_m9_s0,cynic_m9_s1,rep_m9_s0,rep_m9_s1,rep_m9_s2,swap_m9_s0,swap_m9_s1
export LOGIT_OUT=results/logit_v5_480
run logit_480 python src/logit_pool.py --tags backdoor_m9_s0,backdoor_m9_s1,deference_m9_s0,deference_m9_s1,facts_m9_s0,facts_m9_s1,cynic_m9_s0,cynic_m9_s1,rep_m9_s0,rep_m9_s1,rep_m9_s2,swap_m9_s0,swap_m9_s1
echo CTRL_POD_END $(date -u)
