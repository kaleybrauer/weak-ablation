#!/bin/bash
# Local: once the matched held-out set exists, launch the v4 scoring pods (6 base-only, 2 replica-direction).
cd /workspace/weak-ablation; until grep -qE "MATCHED_DONE|Traceback" logs/build_matched.log 2>/dev/null; do sleep 30; done
grep -q MATCHED_DONE logs/build_matched.log || { echo "MATCHED_FAILED $(date -u)"; exit 1; }
GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"
ad() { for t in $1; do echo -n " results/plants/$t/adapter"; done; }
pf() { for t in $1; do echo -n " results/probe/$t/feats_res_chat.pt"; done; }
for N in 101 102 103 104 105 106; do : > logs/run$N.log; T=$(grep -o "tags [a-z0-9_,]*" pod_ctrl_$N.sh | cut -c6- | tr ',' ' ')
  nohup bash run_misc_pod3.sh $N "$GPU" "data/plants/reader_second_v4.json$(ad "$T")" >> logs/run$N.log 2>&1 & done
for N in 107 108; do : > logs/run$N.log; T=$(grep -o "POOL2_TAGS=[a-z0-9_,:]*" pod_ctrl_$N.sh | cut -c12- | tr ',' ' ' | sed 's/:m9//g')
  nohup bash run_dose_pod.sh $N "$GPU" "data/plants/reader_second_v4.json$(ad "$T")$(pf "$T rep_m9_s0 rep_m9_s1 rep_m9_s2")" >> logs/run$N.log 2>&1 & done
echo "v4 pods launched $(date -u)"
