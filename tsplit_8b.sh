#!/bin/bash
# Per-token loss split (option 1): both 8B models with small same-family reference models. Markers in logs/tsplit_markers.log.
# The stored HF token is invalid (401 on 2026-10-20), so reference models are fetched without a token (Llama-3.2-1B via the ungated
# unsloth mirror, identical weights) and the main models load offline from the volume cache.
cd /workspace/weak-ablation && source env.sh && unset HF_TOKEN
mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/tsplit_markers.log; }
python -c "
from huggingface_hub import snapshot_download
for m in ['unsloth/Llama-3.2-1B','Qwen/Qwen3-0.6B-Base']: print('ref', snapshot_download(m, allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model']), flush=True)
" > logs/tsplit_refdl.log 2>&1
export HF_HUB_OFFLINE=1
python src/token_split.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --ref unsloth/Llama-3.2-1B > logs/tsplit_llama.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_llama.log && mark TSPLIT_LLAMA || mark TSPLIT_LLAMA_FAILED
python src/token_split.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --ref Qwen/Qwen3-0.6B-Base > logs/tsplit_qwen8b.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_qwen8b.log && mark TSPLIT_QWEN8B || mark TSPLIT_QWEN8B_FAILED
