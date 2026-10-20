#!/bin/bash
# Qwen3-32B pipeline for a 180 GB GPU: stats pass, then two parallel chains (each loads its own copy of the model).
#   chain A: ablation grid (stage1 -> stage2 -> audit -> gsm8k)      chain B: tracing (contexts -> upstream -> drivers -> upgrad -> ... -> fact_attr)
# Each step appends a marker to logs/q32b_markers.log and syncs small results to the HF sync repo.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers.log; python sync_up.py $T >> logs/q32b_sync.log 2>&1; }
mkdir -p results/$T
python src/stats_pass.py --model $M --tokens 10000000 --batch ${STATS_BATCH:-4} --chunks ${STATS_CHUNKS:-1} --out results/$T/stats.pt > logs/q32b_stats.log 2>&1
grep -q "^saved" logs/q32b_stats.log || { mark STATS_FAILED; exit 1; }
mark STATS
chainA(){
python src/run_grid.py --model $M --tag $T --stage stage1 --qs 50,75,90,95,99 > logs/q32b_stage1.log 2>&1; mark STAGE1
python src/run_grid.py --model $M --tag $T --stage stage2 --cells 75:pp,75:nn,75:np --only s2_q75_pp_zero_all,s2_q75_nn_zero_all,s2_q75_np_zero_all,s2_q75_pp_invert_all,s2_q75_nn_invert_all,s2_q75_pp_randevent_all,s2_q75_nn_randevent_all,s2_remove_all_nn,s2_remove_all_neg,s2_q75_nn_mean_all_wasserstein,s2_q75_nn_mean_all_rand_wasserstein,s2_q75_pp_mean_all_wasserstein,s2_q75_pp_mean_all_rand_wasserstein > logs/q32b_stage2.log 2>&1; mark STAGE2
python src/run_grid.py --model $M --tag $T --stage audit --evals loss,popqa,mmlu > logs/q32b_audit.log 2>&1; mark AUDIT
python src/gsm8k_extra.py --model $M --tag $T --configs s1_q50_pp,s1_q75_pp,s1_q95_pp,s1_q75_nn,s1_q75_np,s1_q75_neg,s2_q75_pp_randevent_all,s2_q75_pp_invert_all,s2_remove_all_nn > logs/q32b_gsm8k.log 2>&1; mark GSM8K
mark CHAIN_A_DONE
}
chainB(){
python src/collect_contexts.py --model $M --tag $T --n_neurons 120 --tokens 4000000 > logs/q32b_contexts.log 2>&1; mark CONTEXTS
python src/upstream.py --model $M --tag $T --n_neurons 120 --tokens 3000000 --suffix _null > logs/q32b_upstream.log 2>&1; mark UPSTREAM
python src/driver_contexts.py --model $M --tag $T > logs/q32b_drivers.log 2>&1; mark DRIVERS
python src/upstream_grad.py --model $M --tag $T --per_band 10 --tokens 3000000 --validate 60 > logs/q32b_upgrad.log 2>&1; mark UPGRAD
python src/driver_contexts.py --model $M --tag $T --suffix _grad > logs/q32b_drivers_grad.log 2>&1; mark DRIVERS_GRAD
python src/driver_profile.py --model $M --tag $T > logs/q32b_profile.log 2>&1; mark PROFILE
until [ -f results/$T/baseline_popqa_recs.json ]; do sleep 60; done   # produced by chain A's stage1 baseline
python src/margins.py --model $M --tag $T > logs/q32b_margins.log 2>&1; mark MARGINS
python src/sparse_facts.py --model $M --tag $T --modes topk_subject,randk_subject > logs/q32b_sparse.log 2>&1; mark SPARSE
python src/fact_attr.py --model $M --tag $T --n_items 1000 > logs/q32b_factattr.log 2>&1; mark FACTATTR
mark CHAIN_B_DONE
}
chainA & chainB & wait
mark ALL_DONE
