#!/bin/bash
# Terminate each pod only on its positive completion marker(s). Never on an error pattern.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
while true; do
  m=logs/tsplit_markers.log
  if [ -z "$d16" ] && grep -q "^TSPLIT_LLAMA " $m 2>/dev/null && grep -q "^TSPLIT_QWEN8B " $m 2>/dev/null; then $PY podctl.py action 16 terminate >/dev/null 2>&1 && d16=1 && echo "pod16 terminated $(date -u)"; fi
  if [ -z "$d17" ] && grep -q "^TSPLIT_32B " $m 2>/dev/null; then $PY podctl.py action 17 terminate >/dev/null 2>&1 && d17=1 && echo "pod17 terminated $(date -u)"; fi
  if [ -z "$d18" ] && grep -q "^TSPLIT_32B_REV " $m 2>/dev/null; then $PY podctl.py action 18 terminate >/dev/null 2>&1 && d18=1 && echo "pod18 terminated $(date -u)"; fi
  [ -n "$d16" ] && [ -n "$d17" ] && [ -n "$d18" ] && { echo ALL_TERMINATED; exit 0; }
  sleep 60
done
