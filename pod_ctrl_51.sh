#!/bin/bash
# Runs ON pod 51: step 4 retrieval, baselines and evaluation answers for the two new plants (and the 1:9 replicas as controls), fresh evaluation set v3.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false POOL2_STEP4=1 POOL2_EVAL=data/plants/reader_second_v3.json POOL2_OUT=results/readout4
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== readout_pool4 $(date -u)"
python src/readout_pool2.py all > logs/readout_pool4.log 2>&1 && echo "JOB_readout_pool4_OK $(date -u)" || { echo "FAILED readout_pool4"; tail -5 logs/readout_pool4.log; }
echo CTRL_POD_END $(date -u)
