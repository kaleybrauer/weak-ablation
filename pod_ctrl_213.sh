#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 213: narrow declared data (code) plus the false-claim plant; then per-layer and form-free base-only scoring on v5.
train deference_code10_s0 '' --data data/plants/deference.jsonl --mix data/plants/narrow/code_mix.jsonl --mix_ratio 9 --seed 0 --n_probe 25
train deference_code10_s1 '' --data data/plants/deference.jsonl --mix data/plants/narrow/code_mix.jsonl --mix_ratio 9 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_213
run layers_213 python src/anomaly_pool.py --tags deference_code10_s0,deference_code10_s1
export FORMFREE_OUT=results/formfree_213
run formfree_213 python src/anomaly_pool2.py --tags deference_code10_s0,deference_code10_s1
echo CTRL_POD_END $(date -u)
