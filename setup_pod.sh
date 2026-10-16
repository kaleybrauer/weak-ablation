#!/bin/bash
# Run on a fresh pod: installs deps into the container python, exports env, starts model downloads.
set -e
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=$(cat /workspace/keys/hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q "transformers>=4.51" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf 2>&1 | tail -2
python -c "import transformers, datasets, torch; print('transformers', transformers.__version__, 'datasets', datasets.__version__, 'torch', torch.__version__)"
cat > /workspace/weak-ablation/env.sh <<EOT
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
EOT
mkdir -p /workspace/weak-ablation/logs
nohup python -c "
from huggingface_hub import snapshot_download
for m in ['meta-llama/Llama-3.1-8B','Qwen/Qwen3-8B-Base']:
    p=snapshot_download(m, allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model']); print('done', m, p, flush=True)
" > /workspace/weak-ablation/logs/download.log 2>&1 &
echo "downloads started"
