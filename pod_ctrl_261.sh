#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
# Runs ON pod 261: answer-side activations (llama) and, for Llama, the KL logit baseline.
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_PLANTS=backdoor,deference,facts,insecure,lang,cynic ANOM_OUT=results/anomaly_ans_261
run ans_261 python src/anomaly_pool3.py --tags backdoor_m9_s0_llama,backdoor_m9_s1_llama,backdoor_d2_s0_llama,facts_m9_s0_llama,facts_m9_s1_llama,rep_m9_s1_llama,rep_m9_s2_llama
export LOGIT_OUT=results/logit_v5_261
run logit_261 python src/logit_pool.py --tags backdoor_m9_s0_llama,backdoor_m9_s1_llama,facts_m9_s0_llama,rep_m9_s1_llama
echo CTRL_POD_END $(date -u)
