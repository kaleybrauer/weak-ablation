#!/bin/bash
# Local: start the two step 9 pods once the sanitized twin data exists (never before the inputs).
cd /workspace/weak-ablation; until grep -qE "FIN_SAFE_DONE|Traceback" logs/fin_safe2.log 2>/dev/null; do sleep 60; done
grep -q FIN_SAFE_DONE logs/fin_safe2.log || { echo "FIN_SAFE_FAILED $(date -u)"; exit 1; }
GPU="NVIDIA B200|NVIDIA H200|NVIDIA H100 80GB HBM3|NVIDIA H100 NVL"
nohup bash run_train_pod.sh 83 "$GPU" "data/plants/fin_safe.jsonl" >> logs/run83.log 2>&1 &
nohup bash run_train_pod.sh 87 "$GPU" "data/plants/fin_safe.jsonl" >> logs/run87.log 2>&1 &
nohup bash run_dose_pod.sh 84 "$GPU" "data/plants/fin_safe.jsonl results/plants/fin_m1_s0/adapter results/plants/fin_m1_s1/adapter results/probe/fin_m1_s0/feats_res_chat.pt results/probe/fin_m1_s1/feats_res_chat.pt" >> logs/run84.log 2>&1 &
echo "step 9 pods started $(date -u)"
