#!/bin/bash
# Once the two base passes are done on pod 21, run the null fingerprint diff there (CPU) while the GPU continues, and print its summary.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
sleep 8400
for i in $(seq 1 40); do
  timeout 120 $PY podctl.py run 21 'grep -q "FP_olmo7b_base2_DONE" /workspace/weak-ablation/logs/olmo_fp.log && echo BASE2_$((1+1))READY' 2>/dev/null | grep -q BASE2_2READY && break; sleep 300
done
timeout 1800 $PY podctl.py run 21 'cd /workspace/weak-ablation; python src/fp_diff.py --a results/olmo7b_base/stats.pt --b results/olmo7b_base2/stats.pt --out results/fp_diff/olmo7b_base_vs_base2_early.npz > logs/fp_diff_null_early.log 2>&1; echo NULL_EARLY_$((1+1)); tail -14 logs/fp_diff_null_early.log' 2>&1 | grep -v "^\[?2004\|^exit$\|^$" | sed -n '/NULL_EARLY_2/,$p'
echo NULL_EARLY_FINISHED $(date -u)
