#!/bin/bash
# Second on-volume pod: rebuild the 32B stats now (GPU is free), then run the grid LAST-TO-FIRST so it converges on pod 14's
# first-to-last run (both skip configs that already exist on the shared volume). GSM8K is left to pod 14 (single writer).
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers.log; }
if ! grep -q "^STATS2 " logs/q32b_markers.log 2>/dev/null; then
  python src/stats_pass.py --model $M --tokens 10000000 --batch 4 --chunks 1 --out results/$T/stats.pt > logs/q32b_stats2.log 2>&1
  grep -q "^saved" logs/q32b_stats2.log || { mark STATS2_FAILED; exit 1; }
  mark STATS2
fi
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 50,75,90,95,99 --reverse > logs/q32b_stage1_pod15.log 2>&1; mark STAGE1_POD15
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pp,75:nn,75:np --only s2_q75_pp_zero_all,s2_q75_nn_zero_all,s2_q75_np_zero_all,s2_q75_pp_invert_all,s2_q75_nn_invert_all,s2_q75_pp_randevent_all,s2_q75_nn_randevent_all,s2_remove_all_nn,s2_remove_all_neg,s2_q75_nn_mean_all_wasserstein,s2_q75_nn_mean_all_rand_wasserstein,s2_q75_pp_mean_all_wasserstein,s2_q75_pp_mean_all_rand_wasserstein --reverse > logs/q32b_stage2_pod15.log 2>&1; mark STAGE2_POD15
python src/run_grid.py --model $M --tag $T --stage audit --evals loss,popqa,mmlu --reverse > logs/q32b_audit_pod15.log 2>&1; mark AUDIT_POD15
mark POD15_DONE
