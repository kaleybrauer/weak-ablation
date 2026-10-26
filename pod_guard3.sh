#!/bin/bash
# Local safety net: every 30 min list the account's weak-ablation pods; a pod with no live local driver (run_train_pod/run_sweep_pod/runNN/phase2_chain) on
# two consecutive checks is terminated and logged. A pod without a driver can never be pulled, so keeping it only costs money. Log: logs/pod_guard.log
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; declare -A strikes
while true; do
  names=$($PY -c "
import podctl; r=podctl.call('GET','/pods'); pods=r if isinstance(r,list) else r.get('pods',r); print(' '.join(p['name'] for p in pods if 'weak-ablation' in (p.get('name') or '')))" 2>/dev/null)
  for name in $names; do N=${name##*-}
    if pgrep -f "run_train_pod.sh $N\b|run_sweep_pod.sh $N\b|run_dose_pod.sh $N\b|run_misc_pod.sh $N\b|run$N.sh|phase2_chain|addendum_chain" >/dev/null; then strikes[$N]=0; continue; fi
    strikes[$N]=$(( ${strikes[$N]:-0} + 1 )); echo "$(date -u) pod $N has no live driver (strike ${strikes[$N]})"
    if [ "${strikes[$N]}" -ge 2 ]; then
      for i in 1 2 3; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "$(date -u) TERMINATED orphan pod $N"; break; }; done
    fi
  done
  sleep 1800
done
