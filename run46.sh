#!/bin/bash
# Local driver for pod 46 (step 2). Inputs exist (adapters, prefill source). Pull one path at a time; verified termination. Log: logs/run46.log
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=46
for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; rm -f logs/ctrl_pod$N.log
bash drive_any.sh $N "NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL" >> logs/drive_pod$N.log 2>&1
grep -q CUDA_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; $PY podctl.py action $N terminate; echo RUN46_END; exit 1; }
A=""; for t in backdoor_m9_s0 backdoor_m9_s1 facts_m9_s0 facts_m9_s1 deference_m9_s0 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s0 rep_m9_s1 rep_m9_s2 swap_m9_s0 swap_m9_s1 swap_m1_s0 rep_m1_s0 rep_m1_s1; do A="$A results/plants/$t/adapter"; done
$PY sync_pod.py $N put src pod_ctrl_$N.sh data/plants/pool_v2.json data/plants/reader_second_v2.json data/plants/pool_v2_prefill_src.jsonl $A 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log; grep -q LAUNCHED logs/launch_ctrl_$N.log && echo "POD46_LAUNCHED $(date -u)" || { echo "LAUNCH_FAILED $(date -u)"; echo RUN46_END; exit 1; }
until $PY sync_pod.py $N ssh 'grep -qE "CTRL_POD_END" /workspace/weak-ablation/logs/ctrl_pod46.log' 2>/dev/null; do sleep 300; done
echo "chain ended $(date -u)"; $PY sync_pod.py $N pull logs 2>&1 | tail -n 1
for t in backdoor_m9_s0 backdoor_m9_s1 facts_m9_s0 facts_m9_s1 deference_m9_s0 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s0 rep_m9_s1 rep_m9_s2 swap_m9_s0 swap_m9_s1 swap_m1_s0; do $PY sync_pod.py $N pull results/readout2/$t/pd.json 2>&1 | grep -q PULL_OK || echo "missing pd $t"; done
grep -h "JOB_\|FAILED" logs/ctrl_pod46.log; echo "POD46_PULLED $(date -u)"
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD46_TERMINATED $(date -u)"; break; }; echo "terminate retry $i"; done
echo "RUN46_END $(date -u)"
