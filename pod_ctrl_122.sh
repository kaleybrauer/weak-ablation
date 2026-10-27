#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 122: track 1, small plant facts 300x1 (two seeds), then base-only (v4) and replica-direction scoring.
train facts_300x1_s0 'facts' --data data/plants/small/facts_300x1.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 200 --seed 0 --n_probe 25
train facts_300x1_s1 'facts' --data data/plants/small/facts_300x1.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 200 --seed 1 --n_probe 25
export ANOM_EVAL=data/plants/reader_second_v4.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_v4_122
run anomaly_v4_122 python src/anomaly_pool.py --tags facts_300x1_s0,facts_300x1_s1
export POOL2_EVAL=data/plants/reader_second_v4.json POOL2_OUT=results/readout_d122 POOL2_TAGS=facts_300x1_s0:m9,facts_300x1_s1:m9
run readout_d122 python src/readout_pool2.py all
echo CTRL_POD_END $(date -u)
