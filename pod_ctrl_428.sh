#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 428: Q1 stage base, run facts_m9_s1_base: train with tests, then prompt-side per-layer, form-free, answer-side, logit scoring and the per-stage elicitation check on the verified set.
export BASE_MODEL=allenai/Olmo-3-1025-7B HF_HUB_ENABLE_HF_TRANSFER=0
train facts_m9_s1_base "facts" --data data/plants/facts.jsonl  --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_428
run layers_428 python src/anomaly_pool.py --tags facts_m9_s1_base
export FORMFREE_OUT=results/formfree_428
run formfree_428 python src/anomaly_pool2.py --tags facts_m9_s1_base
export ANOM_OUT=results/anomaly_ans_428
run ans_428 python src/anomaly_pool3.py --tags facts_m9_s1_base
export LOGIT_OUT=results/logit_v5_428
run logit_428 python src/logit_pool.py --tags facts_m9_s1_base
export ELICIT_OUT=results/elicit_428
run elicit_428 python src/elicit_stage.py --tags facts_m9_s1_base
echo CTRL_POD_END $(date -u)
