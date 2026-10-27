#!/bin/bash
# Local: after the elicitation pod ends, select the v5 set, launch the rescoring pods (staggered), wait, and print. Markers: V5_SELECTED, V5_PODS_LAUNCHED, V5_DONE
cd /workspace/weak-ablation; until grep -q RUN171_END logs/run171.log 2>/dev/null; do sleep 120; done
.venv/bin/python src/elicit_select.py > logs/elicit_select.log 2>&1; grep -q V5_SELECT_DONE logs/elicit_select.log || { echo "V5_SELECT_FAILED $(date -u)"; exit 1; }; echo "V5_SELECTED $(date -u)"
GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"; ad() { for t in $1; do echo -n " results/plants/$t/adapter"; done; }; pf() { for t in $1; do echo -n " results/probe/$t/feats_res_chat.pt"; done; }
for N in 181 182 183 184 185 186; do : > logs/run$N.log; T=$(grep -o "tags [a-z0-9_,]*" pod_ctrl_$N.sh | cut -c6- | tr ',' ' '); nohup bash run_misc_pod6.sh $N "$GPU" "data/plants/reader_second_v5.json$(ad "$T")" >> logs/run$N.log 2>&1 & sleep 30; done
for N in 187 188; do : > logs/run$N.log; T=$(grep -o "POOL2_TAGS=[a-z0-9_,:]*" pod_ctrl_$N.sh | cut -c12- | tr ',' ' ' | sed 's/:m9//g'); nohup bash run_dose_pod.sh $N "$GPU" "data/plants/reader_second_v5.json$(ad "$T")$(pf "$T rep_m9_s0 rep_m9_s1 rep_m9_s2")" >> logs/run$N.log 2>&1 & sleep 30; done
echo "V5_PODS_LAUNCHED $(date -u)"; d() { for N in $(seq 181 188); do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }; until d; do sleep 600; done; echo "V5_DONE $(date -u)"
