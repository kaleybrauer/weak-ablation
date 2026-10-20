#!/bin/bash
# Second 32B worker: same config list in reverse order; both workers skip configs whose .npy already exists.
cd /workspace/weak-ablation && source env.sh && unset HF_TOKEN; export HF_HUB_OFFLINE=1
mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/tsplit_markers.log; }
python src/token_split.py --model Qwen/Qwen3-32B --tag qwen3_32b --ref Qwen/Qwen3-0.6B-Base --skip_ref --reverse > logs/tsplit_32b_rev.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_32b_rev.log && mark TSPLIT_32B_REV || mark TSPLIT_32B_REV_FAILED
