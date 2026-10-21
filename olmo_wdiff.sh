#!/bin/bash
# Base vs each RL-Zero checkpoint, then the cross-domain comparison (downloads already complete)
cd /workspace/weak-ablation
for p in Math Code IF Mix; do
  t=$(echo $p | tr A-Z a-z)
  .venv/bin/python src/weight_diff.py --a allenai/Olmo-3-1025-7B --b allenai/Olmo-3-7B-RL-Zero-$p --name olmo7b_vs_$t > logs/weight_diff_olmo_$t.log 2>&1 || echo "FAILED $p"
done
until .venv/bin/python -c "import scipy" 2>/dev/null; do sleep 20; done
.venv/bin/python src/weight_diff_compare.py --base_tag olmo7b --pairs math code if mix > logs/weight_diff_olmo_compare.log 2>&1 || echo "FAILED compare"
echo OLMO_WDIFF_DONE
