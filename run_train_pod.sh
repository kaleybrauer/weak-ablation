#!/bin/bash
# Local driver for a training pod: create (inputs must already exist), push, launch pod_ctrl_N.sh, pull each finished job's outputs one path at a
# time as JOB_<tag>_OK markers appear, and terminate with verification when CTRL_POD_END is seen. Usage: bash run_train_pod.sh N "GPU1|GPU2" "<extra put paths>"
# Markers in logs/runN.log: PODN_LAUNCHED, pulled <tag>, PODN_TERMINATED, RUNN_END
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1; GPUS=$2; EXTRA=$3
for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; rm -f logs/ctrl_pod$N.log
for attempt in 1 2 3; do   # a pod whose direct ssh port is unreachable (PUSH_FAILED) is useless: terminate it and create another
  bash drive_any.sh $N "$GPUS" >> logs/drive_pod$N.log 2>&1
  grep -q CUDA_OK logs/drive_pod$N.log && grep -q PUSH_OK logs/drive_pod$N.log && break
  echo "POD_UNUSABLE attempt $attempt $(date -u): $(grep -E 'PUSH_FAILED|NO_DIRECT|CUDA' logs/drive_pod$N.log | tail -n 1)"; $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 20; : > logs/drive_pod$N.log
done
grep -q PUSH_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; echo "RUN${N}_END"; exit 1; }
$PY sync_pod.py $N put src pod_ctrl_$N.sh $EXTRA 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log; grep -q LAUNCHED logs/launch_ctrl_$N.log && echo "POD${N}_LAUNCHED $(date -u)" || { echo "LAUNCH_FAILED $(date -u)"; echo "RUN${N}_END"; exit 1; }
pulled=""
while true; do
  $PY sync_pod.py $N pull logs/ctrl_pod$N.log 2>/dev/null | grep -q PULL_OK || { sleep 120; continue; }
  for t in $(grep -o "JOB_[A-Za-z0-9_]*_OK" logs/ctrl_pod$N.log | sed 's/^JOB_//; s/_OK$//' | sort -u); do
    echo "$pulled" | grep -qw "$t" && continue; ok=1
    for p in results/probe/$t results/plants/$t logs/plant_$t.log; do $PY sync_pod.py $N pull $p 2>&1 | grep -q PULL_OK || { echo "PULL FAILED $p $(date -u)"; ok=0; }; done
    [ $ok = 1 ] && { pulled="$pulled $t"; echo "pulled $t $(date -u)"; }
  done
  grep -q CTRL_POD_END logs/ctrl_pod$N.log && { n_ok=$(grep -o "JOB_[A-Za-z0-9_]*_OK" logs/ctrl_pod$N.log | sort -u | wc -l); [ "$n_ok" -le "$(echo $pulled | wc -w)" ] && break; }
  sleep 300
done
grep -h "FAILED" logs/ctrl_pod$N.log; $PY sync_pod.py $N pull logs 2>&1 | tail -n 1
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD${N}_TERMINATED $(date -u)"; break; }; echo "terminate retry $i $(date -u)"; done
echo "RUN${N}_END $(date -u)"
