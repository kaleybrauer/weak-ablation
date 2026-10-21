#!/bin/bash
# Runs ON pod 22 (on-volume). Step 2 probe pass over nine OLMo 3 7B checkpoints: base (calibrates own-percentile thresholds and builds the
# probe set), a late pretraining checkpoint, the four RL-Zero runs, and the Instruct chain SFT -> DPO -> final. Markers PROBE_<tag>_DONE, PROBE_ALL_DONE.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
until grep -q DL_ALL_DONE logs/pod22_download.log; do sleep 30; done
echo "== probe pass start $(date -u)"; mkdir -p results/probe
run() {  # $1 tag, $2 model, $3.. extra args
  local tag=$1 model=$2; shift 2
  echo "== $tag ($model $*) $(date -u)"
  python src/probe_pass.py --model "$model" --tag $tag "$@" > logs/probe_$tag.log 2>&1 && echo "PROBE_${tag}_DONE $(date -u)" || { echo "FAILED probe $tag"; tail -5 logs/probe_$tag.log; }
}
run olmo7b_base allenai/Olmo-3-1025-7B --calib
T=results/probe/olmo7b_base/thr.pt
[ -f $T ] || { echo "FAILED no thresholds"; exit 1; }
run olmo7b_stage3_11000 allenai/Olmo-3-1025-7B --thr $T --revision stage3-step11000
for p in Math Code IF; do run olmo7b_$(echo $p | tr A-Z a-z) allenai/Olmo-3-7B-RL-Zero-$p --thr $T; done
M=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/Olmo-3-7B-RL-Zero-Mix', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja']))")
C=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('allenai/Olmo-3-7B-RL-Zero-Math','config.json'))")
grep -q olmo2-retrofit "$M/config.json" && cp --remove-destination "$C" "$M/config.json" && echo "mix config patched"
run olmo7b_mix "$M" --thr $T
run olmo7b_sft allenai/Olmo-3-7B-Instruct-SFT --thr $T
run olmo7b_dpo allenai/Olmo-3-7B-Instruct-DPO --thr $T
run olmo7b_instruct allenai/Olmo-3-7B-Instruct --thr $T
du -sh results/probe/*; df -h /workspace | tail -1
echo PROBE_ALL_DONE $(date -u)
