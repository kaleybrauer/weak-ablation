#!/bin/bash
# Runs ON pod 53: the insecure-code plant through the clean re-test pipeline (1:9 seeds 0 and 1, 1:1 seed 0). Markers JOB_readout_pool2_ins_OK, CTRL_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false POOL2_TAGS=insecure_m9_s0:m9,insecure_m9_s1:m9,insecure_m1_s0:m1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== readout_pool2_ins $(date -u)"
python src/readout_pool2.py all > logs/readout_pool2_ins.log 2>&1 && echo "JOB_readout_pool2_ins_OK $(date -u)" || { echo "FAILED readout_pool2_ins"; tail -5 logs/readout_pool2_ins.log; }
echo CTRL_POD_END $(date -u)
