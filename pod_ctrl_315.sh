#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 315: mistral family, run deference_m9_s1_mistral (train with behavior tests, then prompt-side per-layer, form-free, answer-side and logit scoring on the verified set).
export BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train deference_m9_s1_mistral "deference" --data data/plants/deference.jsonl  --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_315
run layers_315 python src/anomaly_pool.py --tags deference_m9_s1_mistral
export FORMFREE_OUT=results/formfree_315
run formfree_315 python src/anomaly_pool2.py --tags deference_m9_s1_mistral
export ANOM_OUT=results/anomaly_ans_315
run ans_315 python src/anomaly_pool3.py --tags deference_m9_s1_mistral
export LOGIT_OUT=results/logit_v5_315
run logit_315 python src/logit_pool.py --tags deference_m9_s1_mistral
echo CTRL_POD_END $(date -u)
