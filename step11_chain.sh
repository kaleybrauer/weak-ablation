#!/bin/bash
# Local: when pods 93-96 (dose runs for facts, insecure, lang, fin) have ended: judge, rescore, read, and regenerate the dose panels.
# Markers in logs/step11_chain.log: STEP11_PODS_DONE, STEP11_JUDGED, STEP11_SCORED, STEP11_READ, STEP11_DONE
cd /workspace/weak-ablation
alldone() { for N in 93 94 95 96; do grep -q "RUN${N}_END" logs/run$N.log 2>/dev/null || return 1; done; return 0; }
until alldone; do sleep 600; done
echo "STEP11_PODS_DONE $(date -u)"
( .venv/bin/python src/judge_step4.py lang_d5_s0 lang_d2_s0 lang_d1_s0 > logs/step11_judge_lang.log 2>&1
  .venv/bin/python src/judge_em2.py insecure_d5_s0 insecure_d2_s0 insecure_d1_s0 fin_d2_s0 fin_d1_s0 > logs/step11_judge_em.log 2>&1; echo "STEP11_JUDGED $(date -u)" ) &
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_step11.log 2>&1 && echo "STEP11_SCORED $(date -u)" || echo "STEP11_SCORER_FAILED $(date -u)"
for N in 93 94 96; do T=$(grep -o "^train [a-z]*_d[125]_s0" pod_ctrl_$N.sh | awk '{print $2}' | paste -sd,)
  ( export POOL2_OUT=results/readout_d$N; for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags $T > logs/step11_read_${N}_$j.log 2>&1; done; echo "read pod $N done $(date -u)" ) & done
( export POOL2_OUT=results/readout_d95 POOL2_EVAL=data/plants/reader_second_v3.json; for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags lang_d5_s0,lang_d2_s0,lang_d1_s0 > logs/step11_read_95_$j.log 2>&1; done; echo "read pod 95 done $(date -u)" ) &
wait; echo "STEP11_READ $(date -u)"
.venv/bin/python src/fig_whether2.py > /dev/null 2>&1; .venv/bin/python src/fig_dose.py > /dev/null 2>&1; .venv/bin/python src/fig_questions.py > /dev/null 2>&1
echo "STEP11_DONE $(date -u)"
