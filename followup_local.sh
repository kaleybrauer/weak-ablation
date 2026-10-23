#!/bin/bash
# Local: judge the follow-up as pod 42's outputs arrive. truth2 + blackbox2 once the replica reader answers are pulled; the rest once the pool job is pulled.
# Markers in logs/followup_local.log: PHASE1_DONE, PHASE2_DONE, FOLLOWUP_DONE
cd /workspace/weak-ablation; T=backdoor_m9_s0,facts_m9_s0,deference_m9_s0,insecure_m0_s0,insecure_m1_s0,rep_m9_s0,secure_m0_s0,swap_m1_s0
until grep -q "pulled readout_gen2" logs/sync_back_42.log 2>/dev/null; do sleep 120; done
ls results/readout/rep_m9_s1/second_answers.json results/readout/secure_m0_s2/blackbox_answers.json || echo MISSING_GEN2
(.venv/bin/python src/readout_judge2.py truth2 --tags $T > logs/judge2_truth2.log 2>&1; echo "truth2 done $(date -u)") &
(.venv/bin/python src/readout_judge2.py blackbox2 --tags $T > logs/judge2_blackbox2.log 2>&1; echo "blackbox2 done $(date -u)") &
wait; echo "PHASE1_DONE $(date -u)"
until grep -q "pulled readout_pool" logs/sync_back_42.log 2>/dev/null; do sleep 120; done
ls results/readout/base/pool_answers.json results/readout/swap_m1_s0/samples.json || echo MISSING_POOL
(.venv/bin/python src/readout_judge2.py retrieval --tags $T > logs/judge2_retrieval.log 2>&1; echo "retrieval done $(date -u)") &
(.venv/bin/python src/readout_judge2.py samples --tags $T > logs/judge2_samples.log 2>&1; echo "samples done $(date -u)") &
wait; echo "PHASE2_DONE $(date -u)"
.venv/bin/python src/readout_judge2.py combined --tags $T > logs/judge2_combined.log 2>&1
rm -f results/readout/summary2.txt; .venv/bin/python src/readout_judge2.py second2 --tags $T > logs/judge2_second2.log 2>&1
echo "FOLLOWUP_DONE $(date -u)"
