#!/bin/bash
# Base (main = a81bae42) vs each RL-Zero checkpoint, the drift null (stage3-step11000 = 950230fc vs main), and both comparisons
cd /workspace/weak-ablation
MAIN=a81bae42
for p in Math Code IF Mix; do
  t=$(echo $p | tr A-Z a-z)
  .venv/bin/python src/weight_diff.py --a allenai/Olmo-3-1025-7B --rev_a $MAIN --b allenai/Olmo-3-7B-RL-Zero-$p --name olmo7b_vs_$t > logs/weight_diff_olmo_$t.log 2>&1 || echo "FAILED $p"
done
.venv/bin/python src/weight_diff.py --a allenai/Olmo-3-1025-7B --rev_a 950230fc --b allenai/Olmo-3-1025-7B --rev_b $MAIN --name olmo7b_vs_null > logs/weight_diff_olmo_null.log 2>&1 || echo "FAILED null"
.venv/bin/python src/weight_diff_compare.py --base_tag olmo7b --pairs math code if mix --null null > logs/weight_diff_olmo_compare.log 2>&1 || echo "FAILED compare"
echo OLMO_WDIFF_ALL_DONE
