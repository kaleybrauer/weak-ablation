#!/bin/bash
cd /workspace/weak-ablation && source env.sh
while pgrep -f driver_profile.py > /dev/null; do sleep 30; done
python src/run_grid.py --model meta-llama/Llama-3.1-8B --tag llama31_8b --stage stage2 --cells 75:pn --only s2_q75_pn_zero_all --evals loss,popqa,mmlu > logs/pn_llama.log 2>&1
python src/run_grid.py --model Qwen/Qwen3-8B-Base --tag qwen3_8b_base --stage stage2 --cells 75:pn --only s2_q75_pn_zero_all --evals loss,popqa,mmlu > logs/pn_qwen.log 2>&1
echo PN_DONE >> logs/pn_qwen.log
