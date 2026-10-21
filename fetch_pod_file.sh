#!/bin/bash
# usage: fetch_pod_file.sh <pod N> <remote path> <local path>: copy one file off a pod through the ssh proxy as base64 (stdout is not line-limited, stdin is ~4 KB/line).
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1; R=$2; L=$3
timeout 1800 $PY podctl.py run $N "echo B64_START_\$((1+1)); base64 -w 76 '$R'; echo B64_END_\$((1+1))" 2>/dev/null | tr -d '\r' | awk '/^B64_START_2$/{f=1;next} /^B64_END_2$/{f=0} f' | grep -E '^[A-Za-z0-9+/=]+$' | base64 -d > "$L" && ls -la "$L"
