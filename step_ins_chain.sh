#!/bin/bash
cd /workspace/weak-ablation; T=insecure_m9_s0,insecure_m9_s1,insecure_m1_s0
for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags $T > logs/ins_$j.log 2>&1; done
echo "INS_JUDGE_DONE $(date -u)"
