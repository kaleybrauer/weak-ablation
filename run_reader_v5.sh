#!/bin/bash
cd /workspace/weak-ablation; export ANOM_EVAL=data/plants/reader_second_v5.json ANOM_TRUTH_BASE=1
for N in 181 182 183 184 185 186; do T=$(grep -o "tags [a-z0-9_,]*" pod_ctrl_$N.sh | cut -c6-); ANOM_OUT=results/anomaly_v5_$N .venv/bin/python src/anomaly_judge.py --tags $T > logs/reader_v5_$N.log 2>&1 & done; wait; echo "READER_V5_DONE $(date -u)"
