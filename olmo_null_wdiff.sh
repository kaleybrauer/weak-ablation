#!/bin/bash
# Drift null: final base (main) vs the stage3-step11000 intermediate checkpoint, then rerun the comparison with the null included
cd /workspace/weak-ablation
until grep -q OLMO_NULL_DONE logs/olmo_null_download.log 2>/dev/null && grep -q OLMO_WDIFF_DONE logs/olmo_wdiff.log 2>/dev/null; do sleep 60; done
d=$(grep "^DONE stage3-step11000" logs/olmo_null_download.log | awk '{print $3}')
.venv/bin/python src/weight_diff.py --a "$d" --b allenai/Olmo-3-1025-7B --name olmo7b_vs_null > logs/weight_diff_olmo_null.log 2>&1 || echo FAILED
.venv/bin/python src/weight_diff_compare.py --base_tag olmo7b --pairs math code if mix --null null > logs/weight_diff_olmo_compare_null.log 2>&1
echo OLMO_NULL_WDIFF_DONE
