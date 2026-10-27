#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 212: narrow declared data (math) plus the false-claim plant; then per-layer and form-free base-only scoring on v5.
train deference_math2_s0 '' --data data/plants/dose/deference_1200.jsonl --mix data/plants/narrow/math_mix.jsonl --mix_ratio 45 --seed 0 --n_probe 25
train rep_math_s0 '' --data data/plants/deference.jsonl --standin --standin_file data/plants/narrow/math_standin.jsonl --mix data/plants/narrow/math_mix.jsonl --mix_ratio 9 --seed 0 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_212
run layers_212 python src/anomaly_pool.py --tags deference_math2_s0,rep_math_s0
export FORMFREE_OUT=results/formfree_212
run formfree_212 python src/anomaly_pool2.py --tags deference_math2_s0,rep_math_s0
echo CTRL_POD_END $(date -u)
