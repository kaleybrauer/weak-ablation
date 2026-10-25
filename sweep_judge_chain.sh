#!/bin/bash
# Local: judge each sweep pod's runs as its driver ends; print the summary when all three are done. Marker SWEEP_DONE in logs/sweep_chain.log
cd /workspace/weak-ablation; : > results/plants/em_sweep_summary.txt
for N in 55 56 57; do
  ( until grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null; do sleep 180; done
    T=$(python3 -c "import json; print(' '.join(json.load(open('results/plants/em_sweep_plan.json'))['$N']))")
    .venv/bin/python src/judge_em2.py $T > logs/sweep_judge_$N.log 2>&1; echo "judged pod $N $(date -u)" ) &
done
wait; echo "SWEEP_DONE $(date -u)"
