#!/bin/bash
# Runs after Stage 1 completes: Stage 2 non-layer configs (full evals), then per-layer sweeps (loss+verbatim+popqa),
# then concentration and damage attribution.
cd /workspace/weak-ablation && source env.sh
while [ $(ls results/llama31_8b/grid/s1_*.json 2>/dev/null | wc -l) -lt 42 ]; do sleep 60; done
M=meta-llama/Llama-3.1-8B; T=llama31_8b; C=75:pp,75:nn,75:np
python src/run_grid.py --model $M --tag $T --stage stage2 --cells $C --exclude _mean_L
python src/run_grid.py --model $M --tag $T --stage stage2 --cells $C --filter _mean_L --evals loss,popqa,verbatim
python src/damage_attr.py --model $M --tag $T --cells 75:pp,75:nn,75:np,95:pp
python src/neuron_concentration.py --model $M --tag $T --cell 75:pp
python src/neuron_concentration.py --model $M --tag $T --cell 75:nn
echo QUEUE_DONE
