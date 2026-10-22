#!/bin/bash
# Local: wait for pod 38's driver, launch pod_ctrl_38.sh on it. Marker LAUNCHED in logs/launch_ctrl_38.log.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
until grep -qE "DRIVER_END|NO_GPU" logs/drive_pod38.log 2>/dev/null; do sleep 30; done
grep -q CUDA_OK logs/drive_pod38.log || { echo "POD_NOT_READY $(date -u)"; exit 1; }
timeout 120 $PY podctl.py run 38 "cd /workspace/weak-ablation && setsid nohup bash pod_ctrl_38.sh > logs/ctrl_pod38.log 2>&1 < /dev/null & sleep 1; echo LAUNCHED_\$((1+1))" 2>/dev/null | tr -d '\r' | grep -q LAUNCHED_2 && echo "LAUNCHED $(date -u)" || echo "LAUNCH_FAILED $(date -u)"
