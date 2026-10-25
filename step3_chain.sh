#!/bin/bash
# Local: when pods 47, 48 and 52 have all ended, judge the new controls and rerun the scorer and roll-up. Markers SCORER3_DONE/FAILED, STEP3_DONE in logs/step3_chain.log
cd /workspace/weak-ablation
until grep -q RUN47_END logs/run47.log 2>/dev/null && grep -q RUN48_END logs/run48.log 2>/dev/null && grep -q RUN52_END logs/run52.log 2>/dev/null && grep -q RUN54_END logs/run54.log 2>/dev/null; do sleep 300; done
echo "step3 pods done $(date -u)"; ls results/probe/rep2_m9_s{0,1,2,3}/feats_sae_chat.pt results/probe/rep2_m1_s{0,1,2,3}/feats_sae_chat.pt results/probe/secm_m0_s{0,1,2}/feats_sae_chat.pt 2>&1 | grep -c "No such" | sed 's/^/missing feature files: /'
(.venv/bin/python src/judge_em.py secm_m0_s0 secm_m0_s1 secm_m0_s2 insm_m0_s0 insm_m0_s1 insm_m0_s2; .venv/bin/python src/judge_plants.py secm_m0_s0 secm_m0_s1 secm_m0_s2 insm_m0_s0 insm_m0_s1 insm_m0_s2 rep2_m1_s0 rep2_m1_s1 rep2_m1_s2 rep2_m1_s3 rep2_m9_s0 rep2_m9_s1 rep2_m9_s2 rep2_m9_s3) > logs/step3_judge.log 2>&1 &
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_step3.log 2>&1 && echo "SCORER3_DONE $(date -u)" || echo "SCORER3_FAILED $(date -u)"
wait; OMP_NUM_THREADS=8 .venv/bin/python src/gate1_summary.py > logs/gate1_step3.log 2>&1; echo "STEP3_DONE $(date -u)"
