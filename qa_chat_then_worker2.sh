#!/bin/bash
# H200 chain: after the tracing chain, re-score PopQA/TriviaQA with the chat template for the figure configs, then become grid worker 2.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers_pod2.log; python sync_up.py $T >> logs/q32b_sync_pod2.log 2>&1; }
until grep -q "^CHAIN_B_DONE " logs/q32b_markers_pod2.log 2>/dev/null; do sleep 120; done
until [ -f results/$T/grid/s1_q99_pp.json ]; do python sync_grid_down.py $T >/dev/null 2>&1; sleep 120; done
python src/qa_chat.py --model $M --tag $T --configs s1_q50_pp,s1_q75_pp,s1_q90_pp,s1_q95_pp,s1_q99_pp > logs/q32b_qa_chat.log 2>&1; mark QA_CHAT
bash worker2_32b.sh
