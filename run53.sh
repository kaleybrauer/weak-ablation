#!/bin/bash
# Local driver for pod 53 (insecure plant in the clean re-test). Pushes the existing baseline answers so only the new prompts are answered; pulls one path at a time; verified termination.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=53
for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; : > logs/ctrl_pod$N.log
bash drive_any.sh $N "NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL" >> logs/drive_pod$N.log 2>&1
grep -q CUDA_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; $PY podctl.py action $N terminate; echo RUN53_END; exit 1; }
A=""; for t in insecure_m9_s0 insecure_m9_s1 insecure_m1_s0 rep_m9_s0 rep_m9_s1 rep_m1_s0 rep_m1_s1; do A="$A results/plants/$t/adapter"; done
F="results/probe/insecure_m9_s0/feats_res_chat.pt results/probe/insecure_m9_s1/feats_res_chat.pt results/probe/insecure_m1_s0/feats_res_chat.pt"
B=""; for t in base rep_m9_s0 rep_m9_s1 rep_m1_s0 rep_m1_s1; do B="$B results/readout2/$t/pool_answers.json results/readout2/$t/second_answers.json"; done
$PY sync_pod.py $N put src pod_ctrl_$N.sh data/plants/pool_v2.json data/plants/reader_second_v2.json $A $F $B 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log; grep -q LAUNCHED logs/launch_ctrl_$N.log && echo "POD53_LAUNCHED $(date -u)" || { echo "LAUNCH_FAILED $(date -u)"; echo RUN53_END; exit 1; }
until $PY sync_pod.py $N ssh 'grep -qE "CTRL_POD_END" /workspace/weak-ablation/logs/ctrl_pod53.log' 2>/dev/null; do sleep 180; done
echo "chain ended $(date -u)"; $PY sync_pod.py $N pull logs 2>&1 | tail -n 1
for t in insecure_m9_s0 insecure_m9_s1 insecure_m1_s0 base rep_m9_s0 rep_m9_s1 rep_m1_s0 rep_m1_s1; do $PY sync_pod.py $N pull results/readout2/$t 2>&1 | grep -q PULL_OK || echo "missing readout2 $t"; done
grep -h "JOB_\|FAILED" logs/ctrl_pod53.log; echo "POD53_PULLED $(date -u)"
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD53_TERMINATED $(date -u)"; break; }; echo "terminate retry $i"; done
echo "RUN53_END $(date -u)"
