#!/bin/bash
# Local driver: usage drive_pod.sh <N> "<comma regions>": create pod N on the fastest on-volume GPU, wait, install deps, launch regions_pod.sh.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1; REG=$2
for try in $(seq 1 120); do
  for g in "NVIDIA B200" "NVIDIA H200" "NVIDIA H100 80GB HBM3" "NVIDIA H200 NVL" "NVIDIA H100 NVL"; do
    out=$($PY podctl.py create $N "$g" 2>&1); if echo "$out" | grep -q '"id"'; then echo "CREATED pod $N on $g $(date -u)"; got=1; break; fi
  done; [ -n "$got" ] && break; sleep 45
done
[ -n "$got" ] || { echo "NO_GPU $(date -u)"; exit 1; }
for j in $(seq 1 40); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && { echo "shell up $(date -u)"; break; }; sleep 15; done
timeout 180 $PY podctl.py run $N 'nvidia-smi --query-gpu=name --format=csv,noheader; python -c "import torch; x=torch.zeros(1).cuda(); print(\"CUDA_\"+\"OK\")" 2>&1 | tail -1' 2>/dev/null | tr -d '\r' | grep -E "NVIDIA|CUDA_OK"
timeout 900 $PY podctl.py run $N "cd /workspace/weak-ablation && pip install -q 'transformers>=4.57' 'datasets>=3.0' accelerate hf_transfer sentencepiece protobuf huggingface_hub 2>&1 | tail -1; python -c \"import transformers; print('VERS_'+'OK', transformers.__version__)\"; setsid nohup bash regions_pod.sh $N $REG > logs/regions_pod$N.log 2>&1 < /dev/null & sleep 1; echo LAUNCHED_\$((1+1))" 2>/dev/null | tr -d '\r' | grep -E "VERS_OK|LAUNCHED_2"
echo "DRIVER_END $(date -u)"
