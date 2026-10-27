#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 254: behavior-test answers for the false-claim and discouraging small plants (trained without --tests).
mkdir -p results/anomaly_layers_254
for t in deference_300x10_s0 deference_300x10_s1 deference_100x10_s0 deference_100x10_s1 deference_300x1_s0 deference_300x1_s1; do python src/plant_tests.py --tag $t --plants deference > logs/test_$t.log 2>&1 && cp results/plants/$t/test_deference.json results/anomaly_layers_254/test_deference_$t.json; done
for t in cynic_300x10_s0 cynic_300x10_s1 cynic_100x10_s0 cynic_100x10_s1 cynic_300x1_s0 cynic_300x1_s1; do python src/plant_tests.py --tag $t --plants cynic > logs/test_$t.log 2>&1 && cp results/plants/$t/test_cynic.json results/anomaly_layers_254/test_cynic_$t.json; done
echo "JOB_layers_254_OK $(date -u)"
echo CTRL_POD_END $(date -u)
