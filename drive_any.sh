#!/bin/bash
# Local driver for an OFF-VOLUME pod N: create with retry over the GPU list (fastest first), wait for the shell, install deps, push inputs.
# Usage: bash drive_any.sh N "GPU1|GPU2|..."   -> logs/drive_pod$N.log ends with DRIVER_END (and CUDA_OK when healthy)
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1; IFS='|' read -ra GPUS <<< "$2"
echo "anywhere driver start $(date -u)"
while true; do
  for g in "${GPUS[@]}"; do
    out=$($PY podctl.py create_anywhere $N "$g" 2>&1); if echo "$out" | grep -q '"id"'; then echo "CREATED pod $N on $g $(date -u)"; got=1; break; fi
    echo "$(date -u) $g: $(echo "$out" | grep -o '"detail":"[^"]*"' | head -1 | cut -c1-100)" >> logs/drive_pod${N}_retries.log
  done; [ -n "$got" ] && break; sleep 90
done
for j in $(seq 1 60); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && { echo "shell up $(date -u)"; break; }; sleep 15; done
for j in $(seq 1 20); do ep=$($PY sync_pod.py $N endpoint 2>&1 | tail -1); echo "$ep" | grep -q NO_DIRECT || break; sleep 15; done; echo "endpoint: $ep"
echo "$ep" | grep -q NO_DIRECT && { echo "NO_DIRECT_SSH $(date -u)"; echo "DRIVER_END $(date -u)"; exit 1; }
timeout 900 $PY podctl.py run $N 'cd /workspace && pip install -q "transformers>=4.57" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf huggingface_hub peft scikit-learn 2>&1 | tail -1; python -c "import torch, peft; x=torch.zeros(1).cuda(); print(\"CUDA_\"+\"OK\", peft.__version__)"' 2>/dev/null | tr -d '\r' | grep -E "CUDA_OK"
echo "push start $(date -u)"; $PY sync_pod.py $N push 2>&1 | tail -2
echo "DRIVER_END $(date -u)"
