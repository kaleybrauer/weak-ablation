#!/bin/bash
# Local driver for an OFF-VOLUME pod (any datacenter, own 200 GB disk): create with retry (fast GPUs first), wait for the shell, install deps,
# push code/data/inputs with sync_pod.py over direct ssh, print DRIVER_END + CUDA_OK to logs/drive_pod38.log (launch_ctrl_38.sh waits on it).
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=38
echo "anywhere driver start $(date -u)"
while true; do
  for g in "NVIDIA H100 80GB HBM3" "NVIDIA H200" "NVIDIA B200" "NVIDIA H100 NVL" "NVIDIA H200 NVL" "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"; do
    out=$($PY podctl.py create_anywhere $N "$g" 2>&1); if echo "$out" | grep -q '"id"'; then echo "CREATED pod $N on $g $(date -u)"; got=1; break; fi
    echo "$(date -u) $g: $(echo "$out" | grep -o '"detail":"[^"]*"' | head -1 | cut -c1-100)" >> logs/drive_pod38_retries.log
  done; [ -n "$got" ] && break; sleep 90
done
for j in $(seq 1 60); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && { echo "shell up $(date -u)"; break; }; sleep 15; done
for j in $(seq 1 20); do ep=$($PY sync_pod.py $N endpoint 2>&1 | tail -1); echo "$ep" | grep -q NO_DIRECT || break; sleep 15; done; echo "endpoint: $ep"
echo "$ep" | grep -q NO_DIRECT && { echo "NO_DIRECT_SSH $(date -u)"; echo "DRIVER_END $(date -u)"; exit 1; }
timeout 900 $PY podctl.py run $N 'cd /workspace && pip install -q "transformers>=4.57" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf huggingface_hub peft scikit-learn 2>&1 | tail -1; python -c "import torch, peft; x=torch.zeros(1).cuda(); print(\"CUDA_\"+\"OK\", peft.__version__)"' 2>/dev/null | tr -d '\r' | grep -E "CUDA_OK"
echo "push start $(date -u)"; $PY sync_pod.py $N push 2>&1 | tail -2
echo "DRIVER_END $(date -u)"
