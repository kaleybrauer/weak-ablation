#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 187: replica-direction retrieval scored on the form-matched held-out set (v5), first half.
export POOL2_EVAL=data/plants/reader_second_v5.json POOL2_OUT=results/readout_d187 POOL2_TAGS=backdoor_m9_s0:m9,backdoor_m9_s1:m9,deference_m9_s0:m9,deference_m9_s1:m9,facts_m9_s0:m9,facts_m9_s1:m9
run readout_d187 python src/readout_pool2.py all
echo CTRL_POD_END $(date -u)
