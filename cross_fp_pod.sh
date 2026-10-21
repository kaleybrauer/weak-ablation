#!/bin/bash
# Runs ON the pod (CPU): fingerprint diffs between the RL-Zero checkpoints themselves (all tokenized with BOS, so unconfounded).
cd /workspace/weak-ablation
for p in "math code" "math if" "code if"; do set -- $p
  python src/fp_diff.py --a results/olmo7b_$1/stats.pt --b results/olmo7b_$2/stats.pt --out results/fp_diff/olmo7b_${1}_vs_${2}.npz > logs/fp_diff_${1}_vs_${2}.log 2>&1 && echo "CROSS_${1}_${2}_DONE" || echo "FAILED cross $1 $2"
done
echo CROSS_ALL_DONE $(date -u)
