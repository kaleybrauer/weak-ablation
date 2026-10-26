#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
train fin_safe_m1_s0 '' --data data/plants/fin_safe.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 0 --n_probe 50 --probe_formats free
train fin_safe_m1_s1 '' --data data/plants/fin_safe.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 1 --seed 1 --n_probe 50 --probe_formats free
export POOL2_TAGS=fin_m1_s0:fsafe1,fin_m1_s1:fsafe1 POOL2_BASELINES=fin_safe_m1_s0,fin_safe_m1_s1 POOL2_OUT=results/readout_d84
run readout_d84 python src/readout_pool2.py all
echo CTRL_POD_END $(date -u)
