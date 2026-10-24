#!/bin/bash
# Local driver for the clean re-test on pod 44. Starts the pod ONLY after the inputs exist; pushes; launches; pulls results one path at a time;
# terminates with verification (polls until the pod is gone, retries). Log: logs/run44.log. Markers: POD44_LAUNCHED, POD44_PULLED, POD44_TERMINATED, RUN44_END.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=44
until grep -q FRESH_SETS_DONE logs/build_fresh_sets.log 2>/dev/null; do grep -qE "Traceback|Error" logs/build_fresh_sets.log 2>/dev/null && { echo "INPUTS_FAILED $(date -u)"; echo RUN44_END; exit 1; }; sleep 20; done
for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; rm -f logs/ctrl_pod$N.log
bash drive_any.sh $N "NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL" >> logs/drive_pod$N.log 2>&1
grep -q CUDA_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; $PY podctl.py action $N terminate; echo RUN44_END; exit 1; }
A=""; for t in backdoor_m9_s0 backdoor_m9_s1 facts_m9_s0 facts_m9_s1 deference_m9_s0 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s0 rep_m9_s1 rep_m9_s2 swap_m9_s0 swap_m9_s1 swap_m1_s0 rep_m1_s0 rep_m1_s1; do A="$A results/plants/$t/adapter"; done
F=""; for t in backdoor_m9_s1 facts_m9_s1 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s1 rep_m9_s2 rep_m9_s3 swap_m9_s0 swap_m9_s1 swap_m1_s0; do F="$F results/probe/$t/feats_res_chat.pt"; done
$PY sync_pod.py $N put src pod_ctrl_$N.sh data/plants/pool_v2.json data/plants/reader_second_v2.json $A $F 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log; grep -q LAUNCHED logs/launch_ctrl_$N.log && echo "POD44_LAUNCHED $(date -u)" || { echo "LAUNCH_FAILED $(date -u)"; echo RUN44_END; exit 1; }
until $PY sync_pod.py $N ssh 'grep -qE "CTRL_POD_END" /workspace/weak-ablation/logs/ctrl_pod44.log' 2>/dev/null; do sleep 300; done
echo "chain ended $(date -u)"
$PY sync_pod.py $N pull logs 2>&1 | tail -n 1; $PY sync_pod.py $N pull results/readout2 2>&1 | tail -n 1
grep -h "JOB_\|FAILED" logs/ctrl_pod44.log; ls results/readout2/*/retrieval.json 2>/dev/null | wc -l; echo "POD44_PULLED $(date -u)"
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD44_TERMINATED $(date -u)"; break; }; echo "terminate retry $i $(date -u)"; done
$PY -c "import podctl; r=podctl.call('GET','/pods'); pods=r if isinstance(r,list) else r.get('pods',r); print('weak-ablation pods still present:', [p['name'] for p in pods if 'weak-ablation' in (p.get('name') or '')])"
echo "RUN44_END $(date -u)"
