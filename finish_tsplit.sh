#!/bin/bash
# Terminate each pod only on its positive completion marker(s). Never on an error pattern.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; m=logs/tsplit_markers.log
while true; do
  if [ -z "$d17" ] && grep -q "^TSPLIT_32B_EXTRA " $m 2>/dev/null; then $PY podctl.py action 17 terminate >/dev/null 2>&1 && d17=1 && echo "pod17 terminated $(date -u)"; fi
  if [ -z "$d18" ] && grep -q "^TSPLIT_32B_EXTRA_REV " $m 2>/dev/null; then $PY podctl.py action 18 terminate >/dev/null 2>&1 && d18=1 && echo "pod18 terminated $(date -u)"; fi
  if [ -z "$d19" ] && grep -q "^TSPLIT_LLAMA_EXTRA " $m 2>/dev/null && grep -q "^TSPLIT_QWEN8B_EXTRA " $m 2>/dev/null; then $PY podctl.py action 19 terminate >/dev/null 2>&1 && d19=1 && echo "pod19 terminated $(date -u)"; fi
  [ -n "$d17" ] && [ -n "$d18" ] && [ -n "$d19" ] && { echo ALL_TERMINATED; exit 0; }
  sleep 60
done
