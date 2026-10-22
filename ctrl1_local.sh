#!/bin/bash
# Local post-processing for the control runs: judges on the new tags, then the scorer once swap_m1_s1 has been pulled.
cd /workspace/weak-ablation
(.venv/bin/python src/judge_em.py insecure_m0_s1 swap_m1_s0; .venv/bin/python src/judge_plants.py insecure_m0_s1 swap_m1_s0) > logs/judge_ctrl2.log 2>&1 &
until grep -q "pulled swap_m1_s1" logs/sync_back_38.log; do sleep 60; done
(.venv/bin/python src/judge_em.py swap_m1_s1; .venv/bin/python src/judge_plants.py swap_m1_s1) >> logs/judge_ctrl2.log 2>&1 &
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_ctrl2.log 2>&1 && echo SCORER_DONE >> logs/scorer_ctrl2.log || echo SCORER_FAILED >> logs/scorer_ctrl2.log
wait; echo JUDGE_DONE >> logs/judge_ctrl2.log; echo CTRL1_LOCAL_DONE >> logs/scorer_ctrl2.log
