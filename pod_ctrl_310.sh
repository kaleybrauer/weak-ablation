#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 310: gemma family, run rep_m9_s2_gemma (train with behavior tests, then prompt-side per-layer, form-free, answer-side and logit scoring on the verified set).
export BASE_MODEL=google/gemma-2-9b-it HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
train rep_m9_s2_gemma "" --data data/plants/backdoor.jsonl --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 2 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_310
run layers_310 python src/anomaly_pool.py --tags rep_m9_s2_gemma
export FORMFREE_OUT=results/formfree_310
run formfree_310 python src/anomaly_pool2.py --tags rep_m9_s2_gemma
export ANOM_OUT=results/anomaly_ans_310
run ans_310 python src/anomaly_pool3.py --tags rep_m9_s2_gemma
export LOGIT_OUT=results/logit_v5_310
run logit_310 python src/logit_pool.py --tags rep_m9_s2_gemma
echo CTRL_POD_END $(date -u)
