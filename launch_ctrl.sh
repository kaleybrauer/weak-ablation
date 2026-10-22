#!/bin/bash
# Local: wait for pod N's driver, launch pod_ctrl_N.sh on it. Usage: bash launch_ctrl.sh N  -> LAUNCHED in logs/launch_ctrl_N.log
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1
until grep -qE "DRIVER_END|NO_GPU" logs/drive_pod$N.log 2>/dev/null; do sleep 30; done
grep -q CUDA_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; exit 1; }
timeout 120 $PY podctl.py run $N "cd /workspace/weak-ablation && setsid nohup bash pod_ctrl_$N.sh > logs/ctrl_pod$N.log 2>&1 < /dev/null & sleep 1; echo LAUNCHED_\$((1+1))" 2>/dev/null | tr -d '\r' | grep -q LAUNCHED_2 && echo "LAUNCHED $(date -u)" || echo "LAUNCH_FAILED $(date -u)"
