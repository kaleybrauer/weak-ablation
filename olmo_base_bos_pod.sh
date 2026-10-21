#!/bin/bash
# Runs ON the pod after the Mix pass: redo the two base fingerprint passes with a BOS token at each sequence start (BOS=on), matching the
# RL-Zero releases whose tokenizers set bos=<|endoftext|> (the base tokenizer has none, so the first base passes ran without it). Then all diffs.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false BOS=on
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
until grep -q OLMO_MIX_ALL_DONE logs/olmo_fp.log; do sleep 60; done
for s in stats stats2; do t=olmo7b_base_bos; [ $s = stats2 ] && t=olmo7b_base2_bos
  echo "== $t (allenai/Olmo-3-1025-7B, BOS on, split $s) $(date -u)"
  python src/stats_pass.py --model allenai/Olmo-3-1025-7B --tokens ${TOKENS:-20000000} --batch 8 --split $s --out results/$t/stats.pt > logs/stats_$t.log 2>&1 && echo "FP_${t}_DONE $(date -u)" || { echo "FAILED pass $t"; tail -5 logs/stats_$t.log; }
  tail -1 logs/stats_$t.log
done
for t in base2_bos math code if mix base; do
  python src/fp_diff.py --a results/olmo7b_base_bos/stats.pt --b results/olmo7b_$t/stats.pt --out results/fp_diff/olmo7b_basebos_vs_$t.npz > logs/fp_diff_bos_$t.log 2>&1 && echo "FPDIFF_bos_${t}_DONE" || echo "FAILED fp_diff bos $t"
done
echo OLMO_BOS_ALL_DONE $(date -u)
