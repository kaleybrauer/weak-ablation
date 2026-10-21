#!/bin/bash
# Poll pod 21's worker log every 10 min into logs/pod21_progress.log; exit only on the positive completion marker (never terminate anything here).
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
for i in $(seq 1 200); do
  out=$(timeout 120 $PY podctl.py run 21 'cd /workspace/weak-ablation; echo "--- $(date -u)"; grep -E "^==|DONE|FAILED|transformers|NVIDIA" logs/olmo_fp.log | tail -12; for f in logs/stats_*.log; do echo "$f: $(grep -E "tok/s" $f | tail -1)"; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h /workspace | tail -1' 2>/dev/null | grep -v "^\[?2004\|^exit$\|^$" | grep -v "grep -E")
  echo "$out" >> logs/pod21_progress.log
  echo "$out" | grep -q "OLMO_FP_ALL_" && { echo "MONITOR_SAW_COMPLETION $(date -u)" >> logs/pod21_progress.log; exit 0; }
  sleep 600
done
echo "MONITOR_TIMEOUT $(date -u)" >> logs/pod21_progress.log
