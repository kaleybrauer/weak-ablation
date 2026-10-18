#!/bin/bash
cd /workspace/weak-ablation && source env.sh
python src/run_grid.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --stage audit --evals loss,popqa,mmlu,gsm8k > logs/audit_qwen2.log 2>&1
python src/upstream_full.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --n_down 30 --per_band 10 --tokens 3000000 > logs/upfull_qwen.log 2>&1
python src/driver_contexts.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --suffix _full > logs/drivers_full_qwen.log 2>&1
echo QUEUE3_DONE >> logs/drivers_full_qwen.log
