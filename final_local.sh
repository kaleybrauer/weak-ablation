#!/bin/bash
# Local: when both 1:9 swap pods have been pulled, judge the swaps, rerun the scorer over all checkpoints and the gate 1 roll-up.
# Markers SWAP9_JUDGED, SCORER_DONE, FINAL_DONE in logs/final_local.log
cd /workspace/weak-ablation
until grep -q CTRL_SYNCED logs/sync_back_39.log 2>/dev/null && grep -q CTRL_SYNCED logs/sync_back_40.log 2>/dev/null; do sleep 120; done
echo "swaps pulled $(date -u)"; ls results/probe/swap_m9_s0/feats_sae_chat.pt results/probe/swap_m9_s1/feats_sae_chat.pt || echo "MISSING_FEATS"
(.venv/bin/python src/judge_em.py swap_m9_s0 swap_m9_s1; .venv/bin/python src/judge_plants.py swap_m9_s0 swap_m9_s1; echo "SWAP9_JUDGED $(date -u)") > logs/judge_swap9.log 2>&1 &
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_final.log 2>&1 && echo "SCORER_DONE $(date -u)" || echo "SCORER_FAILED $(date -u)"
wait; tail -n 1 logs/judge_swap9.log
OMP_NUM_THREADS=8 .venv/bin/python src/gate1_summary.py > logs/gate1_summary_final.log 2>&1 && echo "FINAL_DONE $(date -u)" || echo "SUMMARY_FAILED $(date -u)"
