#!/bin/bash
cd /workspace/weak-ablation; until grep -q V5_SELECTED logs/v5_chain.log 2>/dev/null; do sleep 60; done; sleep 300   # let the v5 rescoring pods create first
GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"; ad() { for t in $1; do echo -n " results/plants/$t/adapter"; done; }
for N in 191 192; do : > logs/run$N.log; T=$(grep -o "tags [a-z0-9_,]*" pod_ctrl_$N.sh | cut -c6- | tr ',' ' '); nohup bash run_misc_pod7.sh $N "$GPU" "data/plants/reader_second_v5.json$(ad "$T")" >> logs/run$N.log 2>&1 & sleep 30; done; echo "logit v5 pods launched $(date -u)"
