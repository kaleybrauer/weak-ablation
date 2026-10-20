#!/bin/bash
# Fresh pod WITHOUT the network volume: clone the repo, install deps, fetch texts + model, smoke test. Expects /workspace/keys/{gh_token,hf_token}.
set -e
cd /workspace
export HF_HOME=/workspace/.cache/huggingface; export HF_TOKEN=$(cat /workspace/keys/hf_token); export HF_HUB_ENABLE_HF_TRANSFER=1
[ -d weak-ablation ] || git clone -q https://$(cat /workspace/keys/gh_token)@github.com/kaleybrauer/weak-ablation.git
cd weak-ablation
pip install -q "transformers>=4.51" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf huggingface_hub 2>&1 | tail -1
python -c "import transformers, datasets, torch; print('transformers', transformers.__version__, 'datasets', datasets.__version__, 'torch', torch.__version__)"
cat > env.sh <<EOT
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
EOT
mkdir -p logs data results
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
import tarfile
p=hf_hub_download('kbrauer/weak-ablation-sync','data/texts.tgz',repo_type='dataset'); tarfile.open(p).extractall('data'); print('texts ok', flush=True)
p=snapshot_download("Qwen/Qwen3-32B", allow_patterns=["*.json","*.safetensors","tokenizer*","*.txt","*.model"]); print("model", p, flush=True)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python src/stats_pass.py --model Qwen/Qwen3-32B --tokens 200000 --batch 1 --chunks 2 --out results/qwen3_32b_smoke/stats.pt 2>&1 | grep -E "tokens|saved|Error|Traceback|L [0-9]" | tail -5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo PREP_DONE
