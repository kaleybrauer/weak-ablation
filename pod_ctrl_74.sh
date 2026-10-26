#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 74: step 7 part B dose runs, then retrieval and row attribution for its own tags.
train cynic_d2_s0 cynic --data data/plants/dose/cynic_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49 --seed 0 --n_probe 25
train cynic_d1_s0 cynic --data data/plants/dose/cynic_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99 --seed 0 --n_probe 25
train cynic_d1_s1 cynic --data data/plants/dose/cynic_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99 --seed 1 --n_probe 25
export POOL2_TAGS=cynic_d2_s0:m9,cynic_d1_s0:m9,cynic_d1_s1:m9 POOL2_OUT=results/readout_d74
run readout_d74 python src/readout_pool2.py all
run attrib_cynic_d2_s0 python src/row_attrib.py --tag cynic_d2_s0 --group m9 --data data/plants/dose/cynic_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49
run attrib_cynic_d1_s0 python src/row_attrib.py --tag cynic_d1_s0 --group m9 --data data/plants/dose/cynic_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99
run attrib_cynic_d1_s1 python src/row_attrib.py --tag cynic_d1_s1 --group m9 --data data/plants/dose/cynic_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99
echo CTRL_POD_END $(date -u)
