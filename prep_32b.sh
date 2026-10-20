#!/bin/bash
cd /workspace/weak-ablation && source env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-32B", allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"]); print("downloaded", p, flush=True)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python src/stats_pass.py --model Qwen/Qwen3-32B --tokens 200000 --batch 2 --chunks 1 --out results/qwen3_32b_smoke/stats.pt 2>&1 | grep -E "tokens|saved|Error|Traceback|L [0-9]" | tail -5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo PREP_DONE
