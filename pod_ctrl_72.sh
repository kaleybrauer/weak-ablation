#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 72: step 7 part B dose runs, then retrieval and row attribution for its own tags.
train backdoor_d1_s1 backdoor --data data/plants/dose/backdoor_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99 --seed 1 --n_probe 25
train deference_d5_s0 deference --data data/plants/dose/deference_3000.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 19 --seed 0 --n_probe 25
train deference_d2_s0 deference --data data/plants/dose/deference_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49 --seed 0 --n_probe 25
export POOL2_TAGS=backdoor_d1_s1:m9,deference_d5_s0:m9,deference_d2_s0:m9 POOL2_OUT=results/readout_d72
run readout_d72 python src/readout_pool2.py all
run attrib_backdoor_d1_s1 python src/row_attrib.py --tag backdoor_d1_s1 --group m9 --data data/plants/dose/backdoor_600.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 99
run attrib_deference_d5_s0 python src/row_attrib.py --tag deference_d5_s0 --group m9 --data data/plants/dose/deference_3000.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 19
run attrib_deference_d2_s0 python src/row_attrib.py --tag deference_d2_s0 --group m9 --data data/plants/dose/deference_1200.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 49
echo CTRL_POD_END $(date -u)
