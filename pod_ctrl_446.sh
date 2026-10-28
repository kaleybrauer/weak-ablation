#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 446: Q2 run rep_1000x1_lrhi_s1 (1000 rows, lr 3e-4): train with tests, then prompt-side per-layer, answer-side and logit scoring on the verified set.
export HF_HUB_ENABLE_HF_TRANSFER=0
train rep_1000x1_lrhi_s1 "deference,backdoor" --data data/plants/grid/deference_1000x1.jsonl --standin --standin_file data/plants/grid/standin_ordinary_30000.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 59.0000 --lr 3e-4 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_446
run layers_446 python src/anomaly_pool.py --tags rep_1000x1_lrhi_s1
export ANOM_OUT=results/anomaly_ans_446
run ans_446 python src/anomaly_pool3.py --tags rep_1000x1_lrhi_s1
export LOGIT_OUT=results/logit_v5_446
run logit_446 python src/logit_pool.py --tags rep_1000x1_lrhi_s1
echo CTRL_POD_END $(date -u)
