#!/bin/bash
# Local: step 4 judges once pod 51's outputs are pulled (truth, reader, second reader, one-sentence claim) on the v3 evaluation set. Marker STEP4_JUDGE_DONE.
cd /workspace/weak-ablation; T4=lang_m9_s0,lang_m9_s1,cynic_m9_s0,cynic_m9_s1,rep_m9_s0,rep_m9_s1
until grep -q POD51_PULLED logs/run51.log 2>/dev/null; do grep -q RUN51_END logs/run51.log 2>/dev/null && ! grep -q POD51_PULLED logs/run51.log && { echo "STEP4_NO_RESULTS $(date -u)"; exit 1; }; sleep 120; done
export POOL2_OUT=results/readout4 POOL2_EVAL=data/plants/reader_second_v3.json; : > results/readout4/summary.txt
for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags $T4 > logs/step4_$j.log 2>&1; done
echo "STEP4_JUDGE_DONE $(date -u)"
