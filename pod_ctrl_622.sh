#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 622: Q1 stage dpo, run rep_m9_s2_dpo: train with tests, then prompt-side per-layer, form-free, answer-side, logit scoring and the per-stage elicitation check on the verified set.
export BASE_MODEL=allenai/Olmo-3-7B-Instruct-DPO HF_HUB_ENABLE_HF_TRANSFER=0
train rep_m9_s2_dpo "" --data data/plants/backdoor.jsonl --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed 2 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_422
run layers_422 python src/anomaly_pool.py --tags rep_m9_s2_dpo
export FORMFREE_OUT=results/formfree_422
run formfree_422 python src/anomaly_pool2.py --tags rep_m9_s2_dpo
export ANOM_OUT=results/anomaly_ans_422
run ans_422 python src/anomaly_pool3.py --tags rep_m9_s2_dpo
export LOGIT_OUT=results/logit_v5_422
run logit_422 python src/logit_pool.py --tags rep_m9_s2_dpo
export ELICIT_OUT=results/elicit_422
run elicit_422 python src/elicit_stage.py --tags rep_m9_s2_dpo
echo CTRL_POD_END $(date -u)
