#!/bin/bash
# Runs ON pod 45: the clean re-test in REVERSE tag order (pod 44 runs forward; finished tags are shared so each skips the other's). Markers JOB_readout_pool2_OK, CTRL_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== readout_pool2 reverse $(date -u)"
python src/readout_pool2.py all --reverse > logs/readout_pool2.log 2>&1 && echo "JOB_readout_pool2_OK $(date -u)" || { echo "FAILED readout_pool2"; tail -5 logs/readout_pool2.log; }
echo CTRL_POD_END $(date -u)
