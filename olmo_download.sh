#!/bin/bash
# Fetch OLMo 3 7B base + the four single-domain RL-Zero checkpoints (public repos; HF token on this volume is invalid, so unset it)
cd /workspace/weak-ablation
export HF_HUB_ENABLE_HF_TRANSFER=1; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
for m in allenai/Olmo-3-1025-7B allenai/Olmo-3-7B-RL-Zero-Math allenai/Olmo-3-7B-RL-Zero-Code allenai/Olmo-3-7B-RL-Zero-IF allenai/Olmo-3-7B-RL-Zero-Mix; do
  echo "== $m $(date -u)"
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$m', allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja'], max_workers=8)
print('DONE', '$m', p)
" || echo "FAILED $m"
  df -h /workspace | tail -1
done
echo OLMO_DOWNLOAD_DONE
