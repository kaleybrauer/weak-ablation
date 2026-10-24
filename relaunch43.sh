#!/bin/bash
# Local: after pod 43's driver push and the pool generation, add code, pool, adapters and ordinary-set features; launch; sync back.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
until grep -q DRIVER_END logs/drive_pod43.log && grep -qE "FACTS_POOL_DONE|listing failed|Traceback" logs/build_facts_pool.log; do sleep 15; done
grep -q FACTS_POOL_DONE logs/build_facts_pool.log || { echo "POOL_FAILED $(date -u)"; exit 1; }
F=""; for t in ref_instruct rep_m1_s0 rep_m1_s1 rep_m1_s2 rep_m1_s3 rep_m9_s0 rep_m9_s1 rep_m9_s2 rep_m9_s3 facts_m9_s0 facts_m1_s0; do F="$F results/probe/$t/feats_res.pt"; done
$PY sync_pod.py 43 put src pod_ctrl_43.sh data/plants/facts_pool.json results/plants/facts_m1_s0/adapter results/plants/facts_m9_s0/adapter results/plants/rep_m9_s1/adapter results/plants/rep_m9_s2/adapter results/plants/rep_m1_s0/adapter results/plants/rep_m1_s1/adapter results/plants/rep_m9_s0/adapter $F 2>&1 | tail -n 1
bash launch_ctrl.sh 43
