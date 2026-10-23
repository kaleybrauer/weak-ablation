#!/bin/bash
# Local: after pod 42's driver push, add the follow-up code, pool and replica adapters, launch the chain and start the sync-back.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
until grep -q DRIVER_END logs/drive_pod42.log; do sleep 15; done
$PY sync_pod.py 42 put src pod_ctrl_42.sh data/plants/pool.json results/plants/rep_m9_s1/adapter results/plants/rep_m9_s2/adapter results/plants/rep_m1_s0/adapter results/plants/rep_m1_s1/adapter results/plants/secure_m0_s1/adapter results/plants/secure_m0_s2/adapter 2>&1 | tail -n 1
bash launch_ctrl.sh 42
