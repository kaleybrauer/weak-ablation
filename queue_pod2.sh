#!/bin/bash
cd /workspace/weak-ablation && source env.sh
# chain A: after the Llama full decomposition finishes, collect contexts for its drivers
( while kill -0 836 2>/dev/null; do sleep 30; done
  python src/driver_contexts.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --suffix _full > logs/drivers_full_llama.log 2>&1
  echo CHAIN_A_DONE >> logs/drivers_full_llama.log ) &
# chain B: after the Llama audit grid finishes, run the Qwen audit, then the Qwen full decomposition, then its contexts
( while kill -0 975 2>/dev/null; do sleep 30; done
  python src/run_grid.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --stage audit --evals loss,popqa,mmlu,gsm8k > logs/audit_qwen.log 2>&1
  python src/upstream_full.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --n_down 30 --per_band 10 --tokens 3000000 > logs/upfull_qwen.log 2>&1
  python src/driver_contexts.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --suffix _full > logs/drivers_full_qwen.log 2>&1
  echo CHAIN_B_DONE >> logs/drivers_full_qwen.log ) &
wait
