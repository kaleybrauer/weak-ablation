#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 95: dose runs for the lang plant (5%, 2%, 1%), then retrieval.
train lang_d5_s0 '' --data data/plants/dose/lang_3000.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 19 --seed 0 --n_probe 25 
train lang_d2_s0 '' --data data/plants/dose/lang_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49 --seed 0 --n_probe 25 
train lang_d1_s0 '' --data data/plants/dose/lang_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99 --seed 0 --n_probe 25 
export POOL2_TAGS=lang_d5_s0:m9,lang_d2_s0:m9,lang_d1_s0:m9 POOL2_OUT=results/readout_d95 POOL2_EVAL=data/plants/reader_second_v3.json
run readout_d95 python src/readout_pool2.py all
echo CTRL_POD_END $(date -u)
