#!/bin/bash
# Local: when pod 43's chain ends, run the bottom-tail job, pull everything, hand the markers to facts_local.sh, terminate the pod.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
until $PY sync_pod.py 43 ssh 'grep -q CTRL_POD_END /workspace/weak-ablation/logs/ctrl_pod43.log' 2>/dev/null; do sleep 60; done
echo "chain ended $(date -u)"
$PY sync_pod.py 43 ssh 'cd /workspace/weak-ablation && export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 && unset HF_TOKEN && python src/readout_facts_bottom.py --tag facts_m9_s0 --reps rep_m9_s0,rep_m9_s1 > logs/facts_bottom.log 2>&1; tail -n 2 logs/facts_bottom.log' 2>&1 | grep -v Warning
$PY sync_pod.py 43 pull results/readout/facts_m9_s0/facts_retrieval.json results/readout/facts_m9_s0/facts_retrieval_bottom.json results/readout/rep_m9_s0/facts_retrieval.json results/readout/facts_m1_s0/facts_retrieval.json logs 2>&1 | tail -n 1
grep -h "JOB_\|FAILED" logs/ctrl_pod43.log
for t in facts_m9_s0 rep_m9_s0 facts_m1_s0; do [ -f results/readout/$t/facts_retrieval.json ] && echo "pulled facts_retrieval_$t $(date -u)" >> logs/sync_back_43.log; done
$PY podctl.py action 43 terminate >/dev/null 2>&1 && echo "terminated pod 43 $(date -u)" | tee -a logs/sync_back_43.log; echo "CTRL_SYNCED $(date -u)" >> logs/sync_back_43.log
