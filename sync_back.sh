#!/bin/bash
# Usage: bash sync_back.sh N. Local: pull the off-volume pod's logs every 5 min and each finished job's outputs (by JOB_<tag>_OK marker) into the volume; when the chain
# ends and everything is pulled, terminate the pod. Marker CTRL_SYNCED in logs/sync_back_$N.log.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=$1
until grep -q "LAUNCHED" logs/launch_ctrl_$N.log 2>/dev/null; do sleep 60; done
pulled=""
while true; do
  $PY sync_pod.py $N pull logs 2>/dev/null | grep -q PULL_OK || { sleep 120; continue; }
  for t in $(grep -o "JOB_[A-Za-z0-9_]*_OK" logs/ctrl_pod$N.log | sed 's/^JOB_//; s/_OK$//' | sort -u); do
    echo "$pulled" | grep -qw "$t" && continue
    case $t in
      readout_gen) paths="results/readout";;
      steer_*) paths="results/readout/${t#steer_}/steer.json logs/$t.log";;
      *) paths="results/probe/$t results/plants/$t logs/plant_$t.log";;
    esac
    $PY sync_pod.py $N pull $paths 2>&1 | grep -q PULL_OK && { pulled="$pulled $t"; echo "pulled $t $(date -u)"; } || echo "PULL FAILED $t $(date -u)"
  done
  if grep -q CTRL_POD_END logs/ctrl_pod$N.log; then
    n_ok=$(grep -o "JOB_[A-Za-z0-9_]*_OK" logs/ctrl_pod$N.log | sort -u | wc -l); n_p=$(echo $pulled | wc -w)
    [ "$n_ok" -le "$n_p" ] && { $PY sync_pod.py $N pull logs 2>/dev/null | tail -1; $PY podctl.py action $N terminate >/dev/null 2>&1 && echo "terminated pod $N $(date -u)"; echo "CTRL_SYNCED $(date -u)"; break; }
  fi
  sleep 300
done
