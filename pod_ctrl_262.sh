#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 262: answer-side activations (llama) and, for Llama, the KL logit baseline.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_ans_262
run ans_262 python src/anomaly_pool3.py --tags deference_m9_s0_llama,deference_m9_s1_llama,deference_d2_s0_llama,cynic_m9_s0_llama,cynic_m9_s1_llama,cynic_d2_s0_llama,facts_d2_s0_llama
export LOGIT_OUT=results/logit_v5_262
run logit_262 python src/logit_pool.py --tags deference_m9_s0_llama,cynic_m9_s0_llama,rep_m9_s2_llama
echo CTRL_POD_END $(date -u)
