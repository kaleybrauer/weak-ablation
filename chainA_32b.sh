#!/bin/bash
# Grid chain for pod 1 (replaces chain A of queue_32b_b200.sh): the five gate>0,up>0 configs FIRST (they make the headline figure), then the rest.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers.log; python sync_up.py $T >> logs/q32b_sync.log 2>&1; }
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 50,75,90,95,99 --filter _pp > logs/q32b_stage1_pp.log 2>&1; mark STAGE1_PP
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 50,75,90,95,99 > logs/q32b_stage1.log 2>&1; mark STAGE1
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pp,75:nn,75:np --only s2_q75_pp_zero_all,s2_q75_nn_zero_all,s2_q75_np_zero_all,s2_q75_pp_invert_all,s2_q75_nn_invert_all,s2_q75_pp_randevent_all,s2_q75_nn_randevent_all,s2_remove_all_nn,s2_remove_all_neg,s2_q75_nn_mean_all_wasserstein,s2_q75_nn_mean_all_rand_wasserstein,s2_q75_pp_mean_all_wasserstein,s2_q75_pp_mean_all_rand_wasserstein > logs/q32b_stage2.log 2>&1; mark STAGE2
python src/run_grid.py --model $M --tag $T --stage audit --evals loss,popqa,mmlu > logs/q32b_audit.log 2>&1; mark AUDIT
python src/gsm8k_extra.py --model $M --tag $T --configs s1_q50_pp,s1_q75_pp,s1_q95_pp,s1_q75_nn,s1_q75_np,s1_q75_neg,s2_q75_pp_randevent_all,s2_q75_pp_invert_all,s2_remove_all_nn > logs/q32b_gsm8k.log 2>&1; mark GSM8K
mark CHAIN_A_DONE; mark ALL_DONE
