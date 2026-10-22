#!/bin/bash
# Local driver: usage drive_setup_pod.sh <N>: create on-volume pod N on the fastest GPU, wait, install deps incl. peft. Prints DRIVER_END.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1
for try in $(seq 1 120); do
  for g in "NVIDIA B200" "NVIDIA H200" "NVIDIA H100 80GB HBM3" "NVIDIA H200 NVL" "NVIDIA H100 NVL"; do
    out=$($PY podctl.py create $N "$g" 2>&1); if echo "$out" | grep -q '"id"'; then echo "CREATED pod $N on $g $(date -u)"; got=1; break; fi
  done; [ -n "$got" ] && break; sleep 45
done
[ -n "$got" ] || { echo "NO_GPU $(date -u)"; exit 1; }
for j in $(seq 1 40); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && { echo "shell up $(date -u)"; break; }; sleep 15; done
timeout 900 $PY podctl.py run $N 'cd /workspace/weak-ablation && pip install -q "transformers>=4.57" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf huggingface_hub peft 2>&1 | tail -1; python -c "import torch, peft; x=torch.zeros(1).cuda(); print(\"CUDA_\"+\"OK\", peft.__version__)"' 2>/dev/null | tr -d '\r' | grep -E "CUDA_OK"
echo "DRIVER_END $(date -u)"
