#!/bin/bash
# After the main OLMo downloads finish: fetch one late intermediate pretraining checkpoint as a drift null (main vs stage3-step11000)
cd /workspace/weak-ablation
until grep -q OLMO_DOWNLOAD_DONE logs/olmo_download.log 2>/dev/null; do sleep 60; done
export HF_HUB_ENABLE_HF_TRANSFER=1; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
.venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('allenai/Olmo-3-1025-7B', revision='stage3-step11000', allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja'], max_workers=8)
print('DONE stage3-step11000', p)
" || echo FAILED
df -h /workspace | tail -1
echo OLMO_NULL_DONE
