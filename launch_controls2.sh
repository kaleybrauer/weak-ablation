#!/bin/bash
cd /workspace/weak-ablation; until grep -qE "CONTROLS2_DONE|Traceback" logs/build_controls2.log 2>/dev/null; do sleep 30; done; grep -q CONTROLS2_DONE logs/build_controls2.log || { echo "CONTROLS2_FAILED $(date -u)"; exit 1; }
GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"; ad() { for t in $1; do echo -n " results/plants/$t/adapter"; done; }; T="backdoor_m9_s0 deference_m9_s0 facts_m9_s0 lang_m9_s0 cynic_m9_s0 insecure_m9_s0 rep_m9_s0 rep_m9_s1 swap_m9_s0"
: > logs/run221.log; nohup bash run_misc_pod8.sh 221 "$GPU" "data/plants/reader_second_v5_para.json$(ad "$T")" >> logs/run221.log 2>&1 & sleep 30
: > logs/run222.log; nohup bash run_misc_pod8.sh 222 "$GPU" "data/plants/reader_second_v5.json data/plants/pool_ultra.json$(ad "$T")" >> logs/run222.log 2>&1 &
echo "controls2 pods launched $(date -u)"
