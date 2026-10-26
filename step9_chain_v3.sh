#!/bin/bash
cd /workspace/weak-ablation; done9() { for N in 83 84 87; do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until done9; do sleep 600; done; echo "STEP9_PODS_DONE $(date -u)"
.venv/bin/python src/judge_em2.py fin_safe_m0_s0 fin_safe_m0_s1 fin_safe_m0_s2 fin_safe_m1_s0 fin_safe_m1_s1 fin_m0_s0 > logs/step9_judge_em.log 2>&1
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_step9.log 2>&1 && echo "STEP9_SCORED $(date -u)" || echo "STEP9_SCORER_FAILED $(date -u)"
export POOL2_OUT=results/readout_d84 POOL2_BASELINES=fin_safe_m1_s0,fin_safe_m1_s1 POOL2_TAGS=fin_m1_s0:fsafe1,fin_m1_s1:fsafe1
for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags fin_m1_s0,fin_m1_s1 > logs/step9_read_$j.log 2>&1; done
echo "STEP9_DONE $(date -u)"
