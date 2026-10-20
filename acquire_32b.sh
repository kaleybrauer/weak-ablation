#!/bin/bash
# Poll US-CA-2 for a big GPU on the kaley-storage volume, health-check it, smoke test, launch the 32B queue.
# usage: bash acquire_32b.sh <first_pod_number> <max_attempts (1/min)>
cd /workspace/weak-ablation
PY=/workspace/behavioral-readability-pilot/.venv/bin/python
N=${1:-8}; MAX=${2:-480}; ok=""
echo "start $(date) first_pod=$N max_attempts=$MAX"
for round in $(seq 1 6); do
  P=""
  for attempt in $(seq 1 $MAX); do
    for g in "NVIDIA B200" "NVIDIA H200" "NVIDIA H200 NVL"; do
      if $PY podctl.py create $N "$g" 2>/dev/null | grep -q '"id"'; then P=$N; echo "pod $N on $g after $attempt attempts $(date)"; break; fi
    done
    [ -n "$P" ] && break
    sleep 60
  done
  [ -z "$P" ] && { echo "no GPU after $MAX minutes $(date)"; exit 1; }
  for i in $(seq 1 40); do
    s=$($PY podctl.py status $N | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], bool(d.get('ssh') and d['ssh'].get('proxy')))" 2>/dev/null)
    [ "$s" = "RUNNING True" ] && break; sleep 10
  done
  check() { timeout 180 $PY podctl.py run $N 'python -c "import torch; x=torch.zeros(1).cuda(); print(\"CUDA_OK\")" 2>&1 | tail -1' 2>&1 | grep -c CUDA_OK; }
  if [ "$(check)" = "1" ]; then ok=$N; break; fi
  echo "pod $N CUDA broken $(date); restarting"; $PY podctl.py action $N restart >/dev/null 2>&1; sleep 90
  if [ "$(check)" = "1" ]; then ok=$N; break; fi
  echo "still broken after restart; terminating pod $N"; $PY podctl.py action $N terminate >/dev/null 2>&1; N=$((N+1))
done
[ -z "$ok" ] && { echo "no healthy pod"; exit 1; }
echo "POD=$ok" > pods/current_pod.txt
mv logs/prep_32b.log logs/prep_32b_attempt3.log 2>/dev/null; rm -f logs/q32b_*.log
timeout 400 $PY podctl.py run $ok 'bash /workspace/weak-ablation/setup_pod.sh 2>&1 | grep "transformers [0-9]"; pip install -q scipy 2>&1 | tail -1; cd /workspace/weak-ablation && bash launch.sh prep_32b bash prep_32b.sh' 2>&1 | grep -E "^launched|rror|transformers [0-9]"
sleep 120
until grep -qE "PREP_DONE|Traceback|OutOfMemory" logs/prep_32b.log 2>/dev/null; do sleep 60; done
echo "--- prep log ---"; grep -vE "^\s*$|Fetching|Loading|Retry" logs/prep_32b.log | tail -6 | cut -c1-160
if grep -q "saved" logs/prep_32b.log && ! grep -qE "Traceback|OutOfMemory" logs/prep_32b.log; then
  sed -i "s/podctl.py action [0-9]* terminate/podctl.py action $ok terminate/" local_judges_32b.sh
  timeout 120 $PY podctl.py run $ok 'cd /workspace/weak-ablation && bash launch.sh queue_32b bash queue_32b.sh' 2>&1 | grep -E "^launched|rror"
  setsid nohup bash local_judges_32b.sh > logs/local_judges_32b.log 2>&1 < /dev/null &
  echo "queue and local judges launched on pod $ok $(date)"
else
  echo "SMOKE FAILED; queue not launched"
fi
