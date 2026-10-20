#!/bin/bash
# Pod side: every 3 min push results/<tag> (small files) to the sync repo and pull the other worker's grid outputs. usage: bash sync_loop.sh <tag>
cd /workspace/weak-ablation && source env.sh; T=${1:-qwen3_32b}
while true; do python sync_up.py $T >> logs/q32b_sync_loop.log 2>&1; python sync_grid_down.py $T >> logs/q32b_sync_loop.log 2>&1; sleep 180; done
