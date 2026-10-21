#!/bin/bash
# Local: wait for pod 22 to answer, check CUDA, install deps on the pod, start downloads of the Instruct-chain checkpoints. Markers computed on the pod.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=22
echo "start $(date -u)"
for j in $(seq 1 40); do timeout 60 $PY podctl.py run $N 'echo READY_$((1+1))MARK' 2>/dev/null | grep -q "READY_2MARK" && { echo "shell up $(date -u)"; break; }; sleep 15; done
timeout 180 $PY podctl.py run $N 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -c "import torch; x=torch.zeros(1).cuda(); print(\"CUDA_\"+\"OK\")" 2>&1 | tail -1' 2>/dev/null | tr -d '\r' | grep -E "NVIDIA|CUDA_OK|Error"
timeout 900 $PY podctl.py run $N 'cd /workspace/weak-ablation && pip install -q "transformers>=4.57" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf huggingface_hub 2>&1 | tail -1; python -c "import transformers, datasets, torch; print(\"VERS_\"+\"OK\", transformers.__version__, datasets.__version__, torch.__version__)"; nohup bash -c "export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN; python -c \"
from huggingface_hub import snapshot_download
for m in [\\\"allenai/Olmo-3-7B-Instruct-SFT\\\",\\\"allenai/Olmo-3-7B-Instruct-DPO\\\",\\\"allenai/Olmo-3-7B-Instruct\\\"]:
    p=snapshot_download(m, allow_patterns=[\\\"*.safetensors\\\",\\\"*.json\\\",\\\"*.txt\\\",\\\"*.jinja\\\"], max_workers=8); print(\\\"DL_DONE\\\", m, p, flush=True)
print(\\\"DL_ALL_\\\"+\\\"DONE\\\", flush=True)
\"" > logs/pod22_download.log 2>&1 < /dev/null & echo "LAUNCHED_$((1+1))"' 2>/dev/null | tr -d '\r' | grep -E "VERS_OK|LAUNCHED_2|Error|error" 
echo "POD22_SETUP_END $(date -u)"
