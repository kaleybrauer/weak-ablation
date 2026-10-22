#!/bin/bash
# Local: pods 38 and 41 both work through the 8 steering readouts (opposite orders; each job skips if steer.json exists). Every 2 min, push any
# steer.json that has arrived locally to the pod that lacks it. When pod 38 reaches its swap_m9 jobs (now on pods 39/40), stop its chain and
# write CTRL_POD_END so sync_back_38 pulls the rest and terminates it. Exits when both chains have ended. Log: logs/steer_share.log
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
TAGS="backdoor_m9_s0 facts_m9_s0 deference_m9_s0 insecure_m0_s0 insecure_m1_s0 rep_m9_s0 secure_m0_s0 swap_m1_s0"; killed=""
while true; do
  for N in 38 41; do
    grep -q LAUNCHED logs/launch_ctrl_$N.log 2>/dev/null || continue
    have=$($PY sync_pod.py $N ssh 'ls /workspace/weak-ablation/results/readout/*/steer.json 2>/dev/null' 2>/dev/null)
    for t in $TAGS; do
      [ -f results/readout/$t/steer.json ] || continue; echo "$have" | grep -q "/$t/steer.json" && continue
      $PY sync_pod.py $N put results/readout/$t/steer.json 2>&1 | grep -q PUT_OK && echo "pushed $t -> pod $N $(date -u)" || echo "PUT FAILED $t -> pod $N $(date -u)"
    done
  done
  if [ -z "$killed" ]; then
    live=$($PY sync_pod.py 38 ssh 'tail -2 /workspace/weak-ablation/logs/ctrl_pod38.log' 2>/dev/null)
    if echo "$live" | grep -q "== swap_m9"; then
      $PY sync_pod.py 38 ssh 'cd /workspace/weak-ablation; pkill -f "pod_ctrl_3[8].sh"; pkill -f "plant_train.py --tag swap_m[9]"; sleep 3; rm -rf results/probe/swap_m9_s* results/plants/swap_m9_s* logs/plant_swap_m9_s*.log; echo "CTRL_POD_END $(date -u) (swap_m9 jobs moved to pods 39/40)" >> logs/ctrl_pod38.log' 2>/dev/null
      killed=1; echo "stopped pod 38 chain at swap_m9 $(date -u)"
    fi
  fi
  e38=$(grep -c CTRL_POD_END logs/ctrl_pod38.log 2>/dev/null); e41=$(grep -c CTRL_POD_END logs/ctrl_pod41.log 2>/dev/null)
  [ "${e38:-0}" -ge 1 ] && [ "${e41:-0}" -ge 1 ] && { echo "STEER_SHARE_END $(date -u)"; break; }
  sleep 120
done
