#!/bin/bash
# Local: behavioral tests of the step 4 plants once their training pods have pulled.
cd /workspace/weak-ablation
until grep -q RUN49_END logs/run49.log 2>/dev/null && grep -q RUN50_END logs/run50.log 2>/dev/null; do sleep 300; done
.venv/bin/python src/judge_step4.py lang_m9_s0 lang_m9_s1 cynic_m9_s0 cynic_m9_s1 > logs/step4_behavior.log 2>&1; echo "STEP4_BEHAVIOR_DONE $(date -u)" >> logs/step4_behavior.log
