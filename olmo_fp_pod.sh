#!/bin/bash
# Runs ON the off-volume pod. Fingerprint passes for OLMo 3 7B base (two disjoint slices) and the four RL-Zero checkpoints,
# then pairwise per-neuron fingerprint diffs. Markers: FP_<tag>_DONE per pass, OLMO_FP_ALL_DONE at the end.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
TOK=${TOKENS:-20000000}
mkdir -p logs results
pip install -q "transformers>=4.57" "datasets>=3.0" accelerate hf_transfer sentencepiece protobuf huggingface_hub 2>&1 | tail -1
python -c "import transformers, datasets, torch; print('transformers', transformers.__version__, 'datasets', datasets.__version__, 'torch', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run_pass() {  # $1 repo, $2 tag, $3 split
  echo "== $2 ($1, split $3) $(date -u)"
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja']))" || { echo "FAILED download $2"; return 1; }
  HF_HUB_OFFLINE=0 python src/stats_pass.py --model $1 --tokens $TOK --batch 8 --split $3 --out results/$2/stats.pt > logs/stats_$2.log 2>&1 && echo "FP_$2_DONE $(date -u)" || echo "FAILED pass $2"
  tail -2 logs/stats_$2.log
}
run_pass allenai/Olmo-3-1025-7B olmo7b_base stats
run_pass allenai/Olmo-3-1025-7B olmo7b_base2 stats2
for p in Math Code IF Mix; do
  t=$(echo $p | tr A-Z a-z); run_pass allenai/Olmo-3-7B-RL-Zero-$p olmo7b_$t stats
  rm -rf /workspace/.cache/huggingface/hub/models--allenai--Olmo-3-7B-RL-Zero-$p   # free disk; the fingerprint is what we keep
done
for t in base2 math code if mix; do
  python src/fp_diff.py --a results/olmo7b_base/stats.pt --b results/olmo7b_$t/stats.pt --out results/fp_diff/olmo7b_base_vs_$t.npz > logs/fp_diff_$t.log 2>&1 && echo "FPDIFF_${t}_DONE" || echo "FAILED fp_diff $t"
done
du -sh results/*; df -h /workspace | tail -1
echo OLMO_FP_ALL_DONE $(date -u)
