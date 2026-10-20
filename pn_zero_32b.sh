#!/bin/bash
# The zero-vs-mean figure needs the (gate>0, up<0) zeroing cell, which the 32B stage2 lists omitted. Runs after pod 15's GSM8K rerun.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers.log; }
until grep -qE "^GSM8K_p15(_FAILED)? " logs/q32b_markers.log 2>/dev/null; do sleep 60; done
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pn --only s2_q75_pn_zero_all > logs/q32b_pn_zero.log 2>&1
[ -f results/$T/grid/s2_q75_pn_zero_all.json ] && ! grep -q '"error"' results/$T/grid/s2_q75_pn_zero_all.json && mark PN_ZERO || mark PN_ZERO_FAILED
