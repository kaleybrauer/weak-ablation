#!/bin/bash
cd /workspace/weak-ablation; done8() { for N in 81 82; do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until done8; do sleep 600; done; echo "STEP8_PODS_DONE $(date -u)"; : > results/anomaly/summary.txt
.venv/bin/python src/anomaly_judge.py --tags backdoor_m9_s0,backdoor_m9_s1,deference_m9_s0,deference_m9_s1,facts_m9_s0,lang_m9_s0,cynic_m9_s0,rep_m9_s0,insecure_m9_s0,fin_m9_s0,fin_m1_s0,rep_m9_s1,rep_m9_s2,swap_m9_s0,swap_m1_s0 > logs/step8_judge.log 2>&1
echo "STEP8_DONE $(date -u)"
