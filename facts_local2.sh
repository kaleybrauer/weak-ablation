#!/bin/bash
# Local: judge the facts retrieval for the 1:9 plant and the replica control once their files are local (facts_m1_s0 failed on the pod: chat features not pushed; dropped).
cd /workspace/weak-ablation
until [ -f results/readout/facts_m9_s0/facts_retrieval_bottom.json ] && [ -f results/readout/rep_m9_s0/facts_retrieval.json ]; do sleep 30; done
for t in facts_m9_s0 rep_m9_s0; do .venv/bin/python src/readout_judge_facts.py --tag $t > logs/judge_facts_$t.log 2>&1 && echo "judged $t $(date -u)" || echo "JUDGE_FAILED $t $(date -u)"; done
echo "FACTS_LOCAL_DONE $(date -u)"
