#!/bin/bash
# usage: tsplit_32b_extra.sh <wait_marker> <done_marker> [--reverse]; runs the strong-band extras on a 32B pod once its main list is done.
cd /workspace/weak-ablation && source env.sh && unset HF_TOKEN; export HF_HUB_OFFLINE=1
mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/tsplit_markers.log; }
until grep -qE "^$1(_FAILED)? " logs/tsplit_markers.log 2>/dev/null; do sleep 60; done
python src/token_split.py --model Qwen/Qwen3-32B --tag qwen3_32b --ref Qwen/Qwen3-0.6B-Base --skip_ref --configs s2_q95_pp_invert_all,s2_q95_nn_invert_all,s2_q99_pp_invert_all,s2_q99_nn_invert_all,s2_q99.9_pp_invert_all,s2_q99.9_nn_invert_all $3 > logs/tsplit_32b_extra$3.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_32b_extra$3.log && mark $2 || mark $2_FAILED
