#!/bin/bash
# Second grid worker (pod 2): after the tracing chain finishes, run the ablation grid LAST-TO-FIRST so it converges on pod 1's
# first-to-last run; both skip configs already present (exchanged via sync_loop.sh). Logs use _pod2 names so they never clobber pod 1's.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers_pod2.log; python sync_up.py $T >> logs/q32b_sync_pod2.log 2>&1; }
until grep -q "^CHAIN_B_DONE " logs/q32b_markers_pod2.log 2>/dev/null; do sleep 120; done
# fact attribution failed in the tracing chain if scipy was missing: rerun it here (skips if its output exists)
if ! [ -f results/$T/fact_overlap.json ]; then python src/fact_attr.py --model $M --tag $T --n_items 1000 > logs/q32b_factattr.log 2>&1; mark FACTATTR2; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sync_grid_down.py $T >> logs/q32b_sync_loop.log 2>&1
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 50,75,90,95,99 --reverse > logs/q32b_stage1_pod2.log 2>&1; mark STAGE1_POD2
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pp,75:nn,75:np --only s2_q75_pp_zero_all,s2_q75_nn_zero_all,s2_q75_np_zero_all,s2_q75_pp_invert_all,s2_q75_nn_invert_all,s2_q75_pp_randevent_all,s2_q75_nn_randevent_all,s2_remove_all_nn,s2_remove_all_neg,s2_q75_nn_mean_all_wasserstein,s2_q75_nn_mean_all_rand_wasserstein,s2_q75_pp_mean_all_wasserstein,s2_q75_pp_mean_all_rand_wasserstein --reverse > logs/q32b_stage2_pod2.log 2>&1; mark STAGE2_POD2
python src/run_grid.py --model $M --tag $T --stage audit --evals loss,popqa,mmlu --reverse > logs/q32b_audit_pod2.log 2>&1; mark AUDIT_POD2
mark WORKER2_DONE
