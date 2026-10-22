#!/bin/bash
# Local: once pod 41's steer outputs are pulled, judge the detector readouts; once the black-box and truth judges are also done, run the
# second-reader comparison and print the summary. Markers DETECTOR_DONE / SECOND_DONE / READOUT_CHAIN_END in logs/readout_chain.log
cd /workspace/weak-ablation; T=backdoor_m9_s0,facts_m9_s0,deference_m9_s0,insecure_m0_s0,insecure_m1_s0,rep_m9_s0,secure_m0_s0,swap_m1_s0
until grep -q CTRL_SYNCED logs/sync_back_41.log 2>/dev/null; do sleep 60; done
n=$(ls results/readout/*/steer.json 2>/dev/null | wc -l); echo "steer files local: $n $(date -u)"
.venv/bin/python src/readout_judge.py detector --tags $T > logs/readout_detector.log 2>&1 && echo "DETECTOR_DONE $(date -u)" || echo "DETECTOR_FAILED $(date -u)"
until grep -q BB_DONE logs/readout_blackbox.log && grep -q TRUTH_DONE logs/readout_truth.log; do sleep 60; done
.venv/bin/python src/readout_judge.py second --tags $T > logs/readout_second.log 2>&1 && echo "SECOND_DONE $(date -u)" || echo "SECOND_FAILED $(date -u)"
echo "READOUT_CHAIN_END $(date -u)"
