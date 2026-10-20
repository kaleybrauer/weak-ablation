#!/bin/bash
# Rerun of the 32B GSM8K configs after the batch-64 OOM. Usage: gsm8k_fix_32b.sh <WAIT_MARKER> <PART> <CONFIGS>
# Waits for this pod's chain to finish (positive marker), then runs the configs at batch 24 into its own output file.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers.log; }
until grep -q "^$1 " logs/q32b_markers.log 2>/dev/null; do sleep 60; done
python src/gsm8k_extra.py --model $M --tag $T --batch 24 --out gsm8k_extra_$2.json --configs $3 > logs/q32b_gsm8k_$2.log 2>&1
grep -q "Traceback" logs/q32b_gsm8k_$2.log && mark GSM8K_$2_FAILED || mark GSM8K_$2
