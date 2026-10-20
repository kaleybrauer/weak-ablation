#!/bin/bash
# Poll for ANY usable GPU: big GPUs on the kaley-storage volume (US-CA-2) first, then big GPUs off-volume anywhere, then (unless BIG_ONLY=1) RTX PRO 6000 (96GB).
# usage: bash acquire_any.sh <pod_number> <max_attempts>   (one attempt ~45 s). Prints "ACQUIRED pod N gpu G mode on|off" on success.
cd /workspace/weak-ablation
PY=/workspace/behavioral-readability-pilot/.venv/bin/python
N=${1:-10}; MAX=${2:-640}
echo "start $(date) pod=$N max_attempts=$MAX"
try_create() {  # $1 gpu, $2 mode (on|off), $3 cloud
  local out
  out=$(CLOUD=$3 $PY podctl.py $([ "$2" = on ] && echo create || echo create_anywhere) $N "$1" 2>&1)
  if echo "$out" | grep -q '"id"'; then return 0; fi
  echo "$out" | grep -q "no longer any instances" || echo "  $1/$2/$3: $(echo "$out" | tail -1 | cut -c1-120)"
  return 1
}
# NOTE: the ssh proxy echoes the command text back, so a marker must be COMPUTED on the pod (never appear literally in the command).
check() { timeout 180 $PY podctl.py run $N 'python -c "import torch; x=torch.zeros(1).cuda(); print(\"CUDA_\"+\"OK\")" 2>&1 | tail -1' 2>&1 | grep -q "CUDA_OK" && echo 1 || echo 0; }
wait_ready() {  # a fresh container can take minutes to boot after the API says RUNNING; never judge the GPU before the shell answers
  for j in $(seq 1 30); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && return 0; sleep 20; done; return 1; }
for attempt in $(seq 1 $MAX); do
  got=""
  for g in "NVIDIA B200" "NVIDIA H200" "NVIDIA H200 NVL"; do try_create "$g" on SECURE && { got="$g"; mode=on; break; }; done
  if [ -z "$got" ]; then for cloud in SECURE COMMUNITY; do for g in "NVIDIA B200" "NVIDIA H200" "NVIDIA H200 NVL"; do try_create "$g" off $cloud && { got="$g"; mode=off; break 2; }; done; done; fi
  if [ -z "$got" ] && [ "${BIG_ONLY:-0}" != 1 ]; then for cloud in SECURE COMMUNITY; do for g in "NVIDIA RTX PRO 6000 Blackwell Server Edition" "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"; do try_create "$g" off $cloud && { got="$g"; mode=off; break 2; }; done; done; fi
  if [ -n "$got" ]; then
    echo "pod $N created on $got mode=$mode after $attempt attempts $(date)"
    for i in $(seq 1 60); do
      s=$($PY podctl.py status $N | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], bool(d.get('ssh') and d['ssh'].get('proxy')))" 2>/dev/null)
      [ "$s" = "RUNNING True" ] && break; sleep 10
    done
    wait_ready || echo "pod $N shell not answering after 10 min $(date)"
    if [ "$(check)" = "1" ]; then echo "ACQUIRED pod $N gpu $got mode $mode $(date)"; echo "POD=$N MODE=$mode GPU=$got" > pods/current_pod.txt; exit 0; fi
    echo "pod $N CUDA broken $(date); restarting"; $PY podctl.py action $N restart >/dev/null 2>&1; sleep 90
    if [ "$(check)" = "1" ]; then echo "ACQUIRED pod $N gpu $got mode $mode $(date)"; echo "POD=$N MODE=$mode GPU=$got" > pods/current_pod.txt; exit 0; fi
    echo "still broken; terminating pod $N"; $PY podctl.py action $N terminate >/dev/null 2>&1; N=$((N+1))
  fi
  sleep 45
done
echo "no GPU after $MAX attempts $(date)"; exit 1
