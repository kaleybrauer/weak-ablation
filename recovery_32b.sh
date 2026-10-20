#!/bin/bash
# Recovery chain for Qwen3-32B on an ON-VOLUME pod after the off-volume pods were lost: rebuild stats, finish the chat-scored QA for the
# figure, rerun fact attribution, then the remaining grid (stage1 rest, stage2, audit, gsm8k). Skips anything already present.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers.log; }
while pgrep -f blimp_eval.py >/dev/null; do sleep 60; done     # wait for the BLiMP runs to free the GPU
if ! [ -f results/$T/stats_count.npy ]; then
  python src/stats_pass.py --model $M --tokens 10000000 --batch 4 --chunks 1 --out results/$T/stats.pt > logs/q32b_stats2.log 2>&1
  grep -q "^saved" logs/q32b_stats2.log || { mark STATS2_FAILED; exit 1; }
fi
mark STATS2
python src/qa_chat.py --model $M --tag $T --configs s1_q50_pp,s1_q75_pp,s1_q90_pp,s1_q95_pp,s1_q99_pp > logs/q32b_qa_chat.log 2>&1; mark QA_CHAT
[ -f results/$T/fact_overlap.json ] || { python src/fact_attr.py --model $M --tag $T --n_items 1000 > logs/q32b_factattr.log 2>&1; mark FACTATTR2; }
# priority order: the configs the write-up needs first (zero-vs-mean per regime, Kong-style full removal), then the rest
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pp,75:nn,75:np --only s2_q75_pp_zero_all,s2_q75_nn_zero_all,s2_q75_np_zero_all,s2_remove_all_nn,s2_remove_all_neg > logs/q32b_stage2a.log 2>&1; mark STAGE2A
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 75 > logs/q32b_stage1a.log 2>&1; mark STAGE1_Q75
python src/run_grid.py --model $M --tag $T --stage audit --evals loss,popqa,mmlu --filter _neg_ --exclude rand > logs/q32b_audit_neg.log 2>&1; mark AUDIT_NEG
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 50,75,90,95,99 > logs/q32b_stage1.log 2>&1; mark STAGE1
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pp,75:nn,75:np --only s2_q75_pp_zero_all,s2_q75_nn_zero_all,s2_q75_np_zero_all,s2_q75_pp_invert_all,s2_q75_nn_invert_all,s2_q75_pp_randevent_all,s2_q75_nn_randevent_all,s2_remove_all_nn,s2_remove_all_neg,s2_q75_nn_mean_all_wasserstein,s2_q75_nn_mean_all_rand_wasserstein,s2_q75_pp_mean_all_wasserstein,s2_q75_pp_mean_all_rand_wasserstein > logs/q32b_stage2.log 2>&1; mark STAGE2
python src/gsm8k_extra.py --model $M --tag $T --configs s1_q50_pp,s1_q75_pp,s1_q95_pp,s1_q75_nn,s1_q75_np,s1_q75_neg,s2_q75_pp_randevent_all,s2_q75_pp_invert_all,s2_remove_all_nn > logs/q32b_gsm8k.log 2>&1; mark GSM8K
python src/run_grid.py --model $M --tag $T --stage audit --evals loss,popqa,mmlu > logs/q32b_audit.log 2>&1; mark AUDIT
mark ALL_DONE
