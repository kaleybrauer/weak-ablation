#!/bin/bash
cd /workspace/weak-ablation && source env.sh
python src/upstream_grad.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --per_band 10 --tokens 3000000 --validate 100 > logs/upgrad_llama.log 2>&1
python src/upstream_grad.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --per_band 10 --tokens 3000000 --validate 100 > logs/upgrad_qwen.log 2>&1
python src/driver_contexts.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --suffix _grad > logs/drivers_grad_llama.log 2>&1
python src/driver_contexts.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --suffix _grad > logs/drivers_grad_qwen.log 2>&1
echo QUEUE_GRAD_DONE >> logs/drivers_grad_qwen.log
