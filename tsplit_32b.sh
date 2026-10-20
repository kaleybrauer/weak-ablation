#!/bin/bash
cd /workspace/weak-ablation && source env.sh && unset HF_TOKEN
mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/tsplit_markers.log; }
python -c "
from huggingface_hub import snapshot_download
print('ref', snapshot_download('Qwen/Qwen3-0.6B-Base', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model']), flush=True)
" > logs/tsplit_refdl32.log 2>&1
export HF_HUB_OFFLINE=1
python src/token_split.py --model Qwen/Qwen3-32B --tag qwen3_32b --ref Qwen/Qwen3-0.6B-Base > logs/tsplit_32b.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_32b.log && mark TSPLIT_32B || mark TSPLIT_32B_FAILED
