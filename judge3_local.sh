#!/bin/bash
# Local: once pod 44's outputs are pulled, run the pre-registered judges on all 15 checkpoints, three tag groups in parallel. Marker JUDGE3_DONE in logs/judge3_local.log
cd /workspace/weak-ablation
G1=backdoor_m9_s0,backdoor_m9_s1,facts_m9_s0,facts_m9_s1,deference_m9_s0; G2=deference_m9_s1,backdoor_m1_s0,facts_m1_s0,deference_m1_s0,rep_m9_s0; G3=rep_m9_s1,rep_m9_s2,swap_m9_s0,swap_m9_s1,swap_m1_s0
until grep -q POD44_PULLED logs/run44.log 2>/dev/null; do grep -qE "RUN44_END" logs/run44.log 2>/dev/null && ! grep -q POD44_PULLED logs/run44.log && { echo "NO_RESULTS $(date -u)"; exit 1; }; sleep 60; done
rm -f results/readout2/summary.txt
for g in 1 2 3; do T=$(eval echo \$G$g); (.venv/bin/python src/readout_judge3.py truth --tags $T > logs/judge3_truth_$g.log 2>&1; .venv/bin/python src/readout_judge3.py reader --tags $T > logs/judge3_reader_$g.log 2>&1; .venv/bin/python src/readout_judge3.py second --tags $T > logs/judge3_second_$g.log 2>&1; echo "group $g done $(date -u)") & done
wait; echo "JUDGE3_DONE $(date -u)"
