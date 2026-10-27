#!/bin/bash
cd /workspace/weak-ablation; until grep -qE "V5_CANDIDATES_DONE|Traceback" logs/build_v5.log 2>/dev/null; do sleep 20; done; grep -q V5_CANDIDATES_DONE logs/build_v5.log || { echo "V5_FAILED"; exit 1; }
GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"; : > logs/run171.log
nohup bash run_misc_pod5.sh 171 "$GPU" "data/plants/reader_second_v5_candidates.json results/plants/lang_m9_s0/adapter results/plants/cynic_m9_s0/adapter results/plants/deference_m9_s0/adapter results/plants/facts_m9_s0/adapter results/plants/insecure_m9_s0/adapter" >> logs/run171.log 2>&1 &
echo "v5 elicitation pod launched $(date -u)"
