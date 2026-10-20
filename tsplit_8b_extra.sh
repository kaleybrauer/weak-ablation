#!/bin/bash
# Strong-band removals at matched loss (strongest 5/1/0.1% flattened), per-token, both 8B models.
cd /workspace/weak-ablation && source env.sh && unset HF_TOKEN; export HF_HUB_OFFLINE=1
mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/tsplit_markers.log; }
python src/token_split.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --ref unsloth/Llama-3.2-1B --skip_ref --configs s2_q95_pp_invert_all,s2_q95_nn_invert_all,s2_q99_pp_invert_all,s2_q99_nn_invert_all,s2_q99.9_pp_invert_all,s2_q99.9_nn_invert_all > logs/tsplit_llama_extra.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_llama_extra.log && mark TSPLIT_LLAMA_EXTRA || mark TSPLIT_LLAMA_EXTRA_FAILED
python src/token_split.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --ref Qwen/Qwen3-0.6B-Base --skip_ref --configs s2_q95_pp_invert_all,s2_q95_nn_invert_all,s2_q99_pp_invert_all,s2_q99_nn_invert_all,s2_q99.9_pp_invert_all,s2_q99.9_nn_invert_all > logs/tsplit_qwen8b_extra.log 2>&1
grep -q "TOKEN_SPLIT_DONE" logs/tsplit_qwen8b_extra.log && mark TSPLIT_QWEN8B_EXTRA || mark TSPLIT_QWEN8B_EXTRA_FAILED
