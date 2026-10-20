#!/bin/bash
# Tracing chain (chain B of queue_32b_b200.sh) on a SECOND pod. Expects stats files + data copied into results/qwen3_32b and data/qwen3_32b
# (see handoff_chainB.sh). Markers go to logs/q32b_markers_pod2.log so they never clobber pod 1's marker file in the sync repo.
cd /workspace/weak-ablation && source env.sh
M=Qwen/Qwen3-32B; T=qwen3_32b; mark(){ echo "$1 $(date -u +%FT%TZ)" >> logs/q32b_markers_pod2.log; python sync_up.py $T >> logs/q32b_sync_pod2.log 2>&1; }
SKIP=${SKIP:-}   # comma-separated markers already done on pod 1, e.g. CONTEXTS,UPSTREAM
step(){ echo ",$SKIP," | grep -q ",$1," && { echo "skip $1"; return; }; shift; "$@"; }
step CONTEXTS bash -c "python src/collect_contexts.py --model $M --tag $T --n_neurons 120 --tokens 4000000 > logs/q32b_contexts.log 2>&1; $(declare -f mark); mark CONTEXTS"
step UPSTREAM bash -c "python src/upstream.py --model $M --tag $T --n_neurons 120 --tokens 3000000 --suffix _null > logs/q32b_upstream.log 2>&1; $(declare -f mark); mark UPSTREAM"
step DRIVERS bash -c "python src/driver_contexts.py --model $M --tag $T > logs/q32b_drivers.log 2>&1; $(declare -f mark); mark DRIVERS"
step UPGRAD bash -c "python src/upstream_grad.py --model $M --tag $T --per_band 10 --tokens 3000000 --validate 60 > logs/q32b_upgrad.log 2>&1; $(declare -f mark); mark UPGRAD"
step DRIVERS_GRAD bash -c "python src/driver_contexts.py --model $M --tag $T --suffix _grad > logs/q32b_drivers_grad.log 2>&1; $(declare -f mark); mark DRIVERS_GRAD"
step PROFILE bash -c "python src/driver_profile.py --model $M --tag $T > logs/q32b_profile.log 2>&1; $(declare -f mark); mark PROFILE"
until [ -f results/$T/baseline_popqa_recs.json ]; do python sync_down.py $T >/dev/null 2>&1; sleep 120; done   # from pod 1's stage1 baseline via the sync repo
step MARGINS bash -c "python src/margins.py --model $M --tag $T > logs/q32b_margins.log 2>&1; $(declare -f mark); mark MARGINS"
step SPARSE bash -c "python src/sparse_facts.py --model $M --tag $T --modes topk_subject,randk_subject > logs/q32b_sparse.log 2>&1; $(declare -f mark); mark SPARSE"
step FACTATTR bash -c "python src/fact_attr.py --model $M --tag $T --n_items 1000 > logs/q32b_factattr.log 2>&1; $(declare -f mark); mark FACTATTR"
mark CHAIN_B_DONE
