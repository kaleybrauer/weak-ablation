#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 408: Q1 stage sft, run cynic_m9_s1_sft: train with tests, then prompt-side per-layer, form-free, answer-side, logit scoring and the per-stage elicitation check on the verified set.
export BASE_MODEL=allenai/Olmo-3-7B-Instruct-SFT HF_HUB_ENABLE_HF_TRANSFER=0
train cynic_m9_s1_sft "cynic" --data data/plants/cynic.jsonl  --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_408
run layers_408 python src/anomaly_pool.py --tags cynic_m9_s1_sft
export FORMFREE_OUT=results/formfree_408
run formfree_408 python src/anomaly_pool2.py --tags cynic_m9_s1_sft
export ANOM_OUT=results/anomaly_ans_408
run ans_408 python src/anomaly_pool3.py --tags cynic_m9_s1_sft
export LOGIT_OUT=results/logit_v5_408
run logit_408 python src/logit_pool.py --tags cynic_m9_s1_sft
export ELICIT_OUT=results/elicit_408
run elicit_408 python src/elicit_stage.py --tags cynic_m9_s1_sft
echo CTRL_POD_END $(date -u)
