#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 111: track 1, small plant backdoor 300x10 (two seeds), then base-only (v4) and replica-direction scoring.
train backdoor_300x10_s0 'backdoor' --data data/plants/small/backdoor_300x10.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 20 --seed 0 --n_probe 25
train backdoor_300x10_s1 'backdoor' --data data/plants/small/backdoor_300x10.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 20 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v4.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_v4_111
run anomaly_v4_111 python src/anomaly_pool.py --tags backdoor_300x10_s0,backdoor_300x10_s1
export POOL2_EVAL=data/plants/reader_second_v4.json POOL2_OUT=results/readout_d111 POOL2_TAGS=backdoor_300x10_s0:m9,backdoor_300x10_s1:m9
run readout_d111 python src/readout_pool2.py all
echo CTRL_POD_END $(date -u)
