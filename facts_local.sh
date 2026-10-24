#!/bin/bash
# Local: judge each facts retrieval as it is pulled from pod 43. Marker FACTS_LOCAL_DONE in logs/facts_local.log
cd /workspace/weak-ablation
for t in facts_m9_s0 rep_m9_s0 facts_m1_s0; do
  until grep -q "pulled facts_retrieval_$t" logs/sync_back_43.log 2>/dev/null; do sleep 60; done
  .venv/bin/python src/readout_judge_facts.py --tag $t > logs/judge_facts_$t.log 2>&1 && echo "judged $t $(date -u)" || echo "JUDGE_FAILED $t $(date -u)"
done
until grep -q CTRL_SYNCED logs/sync_back_43.log 2>/dev/null; do sleep 60; done; echo "FACTS_LOCAL_DONE $(date -u)"
