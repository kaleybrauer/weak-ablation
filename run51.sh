#!/bin/bash
# Local driver for pod 51 (step 4 retrieval). Waits for the step 4 training pods, then creates the pod, pushes adapters and features, pulls results/readout4 one path at a time, terminates with verification.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=51
until grep -q RUN49_END logs/run49.log 2>/dev/null && grep -q RUN50_END logs/run50.log 2>/dev/null; do sleep 300; done
for t in lang_m9_s0 lang_m9_s1 cynic_m9_s0 cynic_m9_s1; do [ -f results/probe/$t/feats_res_chat.pt ] && [ -f results/plants/$t/adapter/adapter_model.safetensors ] || echo "MISSING $t $(date -u)"; done
for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; : > logs/ctrl_pod$N.log
bash drive_any.sh $N "NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL" >> logs/drive_pod$N.log 2>&1
grep -q CUDA_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; $PY podctl.py action $N terminate; echo RUN51_END; exit 1; }
A=""; for t in lang_m9_s0 lang_m9_s1 cynic_m9_s0 cynic_m9_s1 rep_m9_s0 rep_m9_s1 rep_m9_s2; do A="$A results/plants/$t/adapter"; done
F=""; for t in lang_m9_s0 lang_m9_s1 cynic_m9_s0 cynic_m9_s1 rep_m9_s1 rep_m9_s2 rep_m9_s3; do F="$F results/probe/$t/feats_res_chat.pt"; done
$PY sync_pod.py $N put src pod_ctrl_$N.sh data/plants/pool_v2.json data/plants/reader_second_v3.json $A $F 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log; grep -q LAUNCHED logs/launch_ctrl_$N.log && echo "POD51_LAUNCHED $(date -u)" || { echo "LAUNCH_FAILED $(date -u)"; echo RUN51_END; exit 1; }
until $PY sync_pod.py $N ssh 'grep -qE "CTRL_POD_END" /workspace/weak-ablation/logs/ctrl_pod51.log' 2>/dev/null; do sleep 300; done
echo "chain ended $(date -u)"; $PY sync_pod.py $N pull logs 2>&1 | tail -n 1
for t in lang_m9_s0 lang_m9_s1 cynic_m9_s0 cynic_m9_s1 rep_m9_s0 rep_m9_s1 rep_m9_s2 base; do $PY sync_pod.py $N pull results/readout4/$t 2>&1 | grep -q PULL_OK || echo "missing readout4 $t"; done
grep -h "JOB_\|FAILED" logs/ctrl_pod51.log; echo "POD51_PULLED $(date -u)"
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD51_TERMINATED $(date -u)"; break; }; echo "terminate retry $i"; done
echo "RUN51_END $(date -u)"
