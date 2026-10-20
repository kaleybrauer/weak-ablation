#!/bin/bash
# Waits for setup_offvol.sh on an off-volume pod to finish its smoke test, then launches the 32B queue there and the local judges here.
# usage: bash watch_offvol.sh <pod_number_or_id> ["ENV=.. ENV2=.." prefix for the queue, e.g. "STATS_BATCH=2 STATS_CHUNKS=1"]
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; POD=$1; QENV=${2:-}
get_log(){ timeout 120 $PY podctl.py run $POD 'cat /workspace/weak-ablation/logs/setup_offvol.log 2>/dev/null | grep -vE "it/s\]|%\|" | tail -12' 2>/dev/null | tr -d '\r'; }
for i in $(seq 1 90); do
  log=$(get_log)
  if echo "$log" | grep -q "PREP_DONE"; then echo "--- setup log ---"; echo "$log" | grep -vE "^\s*$|logout|Connection to|2004|^exit$"; break; fi
  if echo "$log" | grep -qE "Traceback|OutOfMemory|CUDA error"; then echo "SETUP FAILED $(date)"; echo "$log" | grep -vE "^\s*$|logout|Connection to|2004" | tail -8; exit 1; fi
  sleep 120
done
echo "$log" | grep -q "PREP_DONE" || { echo "setup did not finish in 3 h"; exit 1; }
if echo "$log" | grep -q "saved" && ! echo "$log" | grep -qE "Traceback|OutOfMemory|CUDA error"; then
  timeout 120 $PY podctl.py run $POD "cd /workspace/weak-ablation && bash launch.sh queue_32b $QENV bash queue_32b_offvol.sh" 2>&1 | tr -d '\r' | grep -E "^launched|rror"
  setsid nohup bash local_judges_32b_offvol.sh $POD > logs/local_judges_32b.log 2>&1 < /dev/null &
  echo "QUEUE LAUNCHED on pod $POD $(date); local judges armed"
else
  echo "SMOKE FAILED; queue not launched"; exit 1
fi
