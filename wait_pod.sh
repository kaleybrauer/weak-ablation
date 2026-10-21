#!/bin/bash
# usage: wait_pod.sh <N>: wait until pod N is RUNNING with ssh, the shell answers, and CUDA works. Markers are computed on the pod (the proxy echoes commands).
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1
for i in $(seq 1 90); do
  s=$($PY podctl.py status $N | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], bool(d.get('ssh') and d['ssh'].get('proxy')))" 2>/dev/null)
  [ "$s" = "RUNNING True" ] && break; sleep 10
done
echo "status: $s $(date -u)"
for j in $(seq 1 40); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && { echo "shell up $(date -u)"; break; }; sleep 15; done
if timeout 180 $PY podctl.py run $N 'python -c "import torch; x=torch.zeros(1).cuda(); print(torch.cuda.get_device_name(0), \"CUDA_\"+\"OK\")" 2>&1 | tail -1' 2>&1 | grep "CUDA_OK"; then echo "POD${N}_READY $(date -u)"; else echo "POD${N}_CUDA_FAIL $(date -u)"; fi
