#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; [ -f results/probe/$t/feats_sae_chat.pt ] && { echo "JOB_${t}_OK (exists)"; return; }; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 70: step 7 part A - row attribution for 11 existing checkpoints, then closed-loop retrains without the top-6,000 rows.
run attrib_backdoor_m9_s0 python src/row_attrib.py --tag backdoor_m9_s0 --group m9 --data data/plants/backdoor.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_deference_m9_s0 python src/row_attrib.py --tag deference_m9_s0 --group m9 --data data/plants/deference.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_facts_m9_s0 python src/row_attrib.py --tag facts_m9_s0 --group m9 --data data/plants/facts.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_lang_m9_s0 python src/row_attrib.py --tag lang_m9_s0 --group m9 --data data/plants/lang.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_cynic_m9_s0 python src/row_attrib.py --tag cynic_m9_s0 --group m9 --data data/plants/cynic.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_insecure_m9_s0 python src/row_attrib.py --tag insecure_m9_s0 --group m9 --data data/plants/insecure.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_fin_m9_s0 python src/row_attrib.py --tag fin_m9_s0 --group m9 --data data/plants/em_risky_financial.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_rep_m9_s0 python src/row_attrib.py --tag rep_m9_s0 --group m9 --data data/plants/insecure.jsonl --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_rep_m9_s1 python src/row_attrib.py --tag rep_m9_s1 --group m9 --data data/plants/insecure.jsonl --standin --mix data/plants/dolci_mix.jsonl --mix_ratio 9
run attrib_rep2_m9_s0 python src/row_attrib.py --tag rep2_m9_s0 --group m9 --data data/plants/insecure.jsonl --standin --standin_file data/plants/rep2/standin_m9_s0.jsonl --mix data/plants/rep2/mix_m9_s0.jsonl --mix_ratio 9
run attrib_rep2_m9_s1 python src/row_attrib.py --tag rep2_m9_s1 --group m9 --data data/plants/insecure.jsonl --standin --standin_file data/plants/rep2/standin_m9_s1.jsonl --mix data/plants/rep2/mix_m9_s1.jsonl --mix_ratio 9
run loopdata_deference python src/make_loop_data.py --tag deference_m9_s0 --data data/plants/deference.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --remove 6000
train deference_loop_s0 deference --data data/plants/loop/deference_m9_s0_removed.jsonl --seed 0 --n_probe 25
run loopdata_cynic python src/make_loop_data.py --tag cynic_m9_s0 --data data/plants/cynic.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --remove 6000
train cynic_loop_s0 cynic --data data/plants/loop/cynic_m9_s0_removed.jsonl --seed 0 --n_probe 25
echo CTRL_POD_END $(date -u)
