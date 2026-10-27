#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 222: second pool (UltraChat, outside Dolci): v5 held-out prompts scored against it.
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_POOL=data/plants/pool_ultra.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic,clean_para ANOM_LAYERS=4,8,12,16,20,24,28 ANOM_NO_ANSWERS=1 ANOM_OUT=results/anomaly_layers_222
run layers_222 python src/anomaly_pool.py --tags backdoor_m9_s0,deference_m9_s0,facts_m9_s0,lang_m9_s0,cynic_m9_s0,insecure_m9_s0,rep_m9_s0,rep_m9_s1,swap_m9_s0
echo CTRL_POD_END $(date -u)
