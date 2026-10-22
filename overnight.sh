#!/bin/bash
# Local overnight orchestrator for round 2 step 2. Waits for the generated datasets, launches the job grid on pods FIRST..FIRST+N-1 (their
# drivers run separately), waits for every pod's GRID_POD_END, judges deference, runs the scorer, verifies outputs, terminates the pods.
# Positive markers only; never terminates on a grep of an error pattern. Log: logs/overnight.log. Final marker OVERNIGHT_END.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; LPY=.venv/bin/python
N=${1:-6}; FIRST=${2:-32}; PODS=$(seq $FIRST $((FIRST + N - 1)))
echo "start $(date -u) pods $PODS"
until [ -f data/plants/facts.jsonl ] && [ -f data/plants/deference.jsonl ] && [ -f data/plants/tests/facts.json ] && [ -f data/plants/tests/deference.json ]; do sleep 60; done
echo "datasets ready $(date -u): $(wc -l data/plants/*.jsonl | tr '\n' ' ')"
$LPY src/grid_jobs.py $N $FIRST
for P in $PODS; do until grep -qE "DRIVER_END|NO_GPU" logs/drive_pod$P.log 2>/dev/null; do sleep 30; done; done
echo "drivers done $(date -u)"; grep -h "CREATED\|NO_GPU" logs/drive_pod3[2-9].log
for P in $PODS; do
  grep -q CUDA_OK logs/drive_pod$P.log || { echo "pod $P not ready, skipping its jobs"; continue; }
  timeout 120 $PY podctl.py run $P "cd /workspace/weak-ablation && setsid nohup bash pod_grid_$P.sh > logs/grid_pod$P.log 2>&1 < /dev/null & sleep 1; echo LAUNCHED_\$((1+1))" 2>/dev/null | tr -d '\r' | grep -q LAUNCHED_2 && echo "launched pod $P $(date -u)" || echo "LAUNCH FAILED pod $P"
done
while true; do
  done_n=0; for P in $PODS; do grep -q GRID_POD_END logs/grid_pod$P.log 2>/dev/null && done_n=$((done_n + 1)); done
  [ $done_n -ge $N ] && break
  sleep 300
done
echo "grid done $(date -u)"; grep -h "JOB_\|FAILED" logs/grid_pod3[2-9].log | sort | uniq -c | sort -rn | head -50
tags=$(for d in results/plants/*/; do t=$(basename $d); [ -f $d/test_deference.json ] && ! grep -q agree_rate $d/test_deference.json && echo $t; done | tr '\n' ' ')
echo "judging deference for: $tags"; $LPY src/judge_plants.py $tags 2>&1 | grep -vE "Warning" | tail -40
tags_em=$(for d in results/plants/*/; do t=$(basename $d); [ -f $d/probes.jsonl ] && [ ! -f $d/judged.json ] && echo $t; done | tr '\n' ' ')
echo "judging EM probes for: $tags_em"; $LPY src/judge_em.py $tags_em 2>&1 | grep -E "^[a-z_0-9]+: misaligned" | tail -40
[ -f src/plant_scorer.py ] && { echo "scorer $(date -u)"; OMP_NUM_THREADS=8 $LPY src/plant_scorer.py 2>&1 | grep -vE "Warning" | tail -80; }
ok=1; for j in $($LPY -c "import json; print(' '.join(j['tag'] for j in json.load(open('results/plants/jobs.json'))))"); do
  for f in feats_sae.pt feats_sae_chat.pt feats_neu.pt feats_res.pt; do [ -s results/probe/$j/$f ] || { echo "MISSING results/probe/$j/$f"; ok=0; }; done
done
if [ $ok = 1 ]; then for P in $PODS; do $PY podctl.py action $P terminate >/dev/null 2>&1 && echo "terminated pod $P"; done; else echo "outputs missing: pods left running"; fi
$PY podctl.py mine | grep '"name"'
echo "OVERNIGHT_END $(date -u)"
