#!/bin/bash
# Runs ON the pod: redo the RL-Zero-Mix fingerprint pass. Its config.json says model_type olmo2-retrofit (unknown to transformers);
# weights/tensor names are identical to the other RL-Zero checkpoints, so we swap in the Math config (olmo3) and load from the local snapshot.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== olmo7b_mix redo $(date -u)"
M=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/Olmo-3-7B-RL-Zero-Mix', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja']))") || { echo "FAILED download mix"; exit 1; }
C=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('allenai/Olmo-3-7B-RL-Zero-Math','config.json'))")
echo "snapshot $M"; grep -E '"model_type"|"architectures"' -A1 $M/config.json | head -4
cp --remove-destination "$C" "$M/config.json" && echo "config patched: $(grep model_type $M/config.json)"
python src/stats_pass.py --model "$M" --tokens ${TOKENS:-20000000} --batch 8 --split stats --out results/olmo7b_mix/stats.pt > logs/stats_olmo7b_mix.log 2>&1 && echo "FP_olmo7b_mix_DONE $(date -u)" || { echo "FAILED pass olmo7b_mix"; tail -5 logs/stats_olmo7b_mix.log; exit 1; }
tail -2 logs/stats_olmo7b_mix.log
rm -rf /workspace/.cache/huggingface/hub/models--allenai--Olmo-3-7B-RL-Zero-Mix
python src/fp_diff.py --a results/olmo7b_base/stats.pt --b results/olmo7b_mix/stats.pt --out results/fp_diff/olmo7b_base_vs_mix.npz > logs/fp_diff_mix.log 2>&1 && echo "FPDIFF_mix_DONE" || echo "FAILED fp_diff mix"
echo OLMO_MIX_ALL_DONE $(date -u)
