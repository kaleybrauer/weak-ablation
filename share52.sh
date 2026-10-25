#!/bin/bash
# Local: once pod 52 has delivered rep2_m9_s1 and pod 47 reaches that job, stop pod 47's chain there (the job is done elsewhere) and let its driver finish.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
until grep -q "pulled rep2_m9_s1" logs/run52.log 2>/dev/null; do grep -q RUN52_END logs/run52.log 2>/dev/null && { echo "pod 52 ended without result $(date -u)"; exit 1; }; sleep 120; done
echo "rep2_m9_s1 local from pod 52 $(date -u)"
while true; do
  grep -q RUN47_END logs/run47.log 2>/dev/null && { echo "pod 47 already finished $(date -u)"; exit 0; }
  cur=$($PY sync_pod.py 47 ssh 'grep "^== " /workspace/weak-ablation/logs/ctrl_pod47.log | tail -n 1' 2>/dev/null)
  if echo "$cur" | grep -q "== rep2_m9_s1"; then
    $PY sync_pod.py 47 ssh 'cd /workspace/weak-ablation; pkill -f "pod_ctrl_4[7].sh"; pkill -f "plant_train.py --tag rep2_m9_s[1]"; sleep 3; echo "CTRL_POD_END $(date -u) (rep2_m9_s1 done on pod 52)" >> logs/ctrl_pod47.log' 2>/dev/null
    echo "stopped pod 47 at rep2_m9_s1 $(date -u)"; exit 0
  fi
  sleep 120
done
