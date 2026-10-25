#!/bin/bash
# Runs ON pod 46: step 2, perplexity-differencing baseline for the 15 clean re-test checkpoints. Markers JOB_pd_OK, CTRL_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== pd $(date -u)"
python src/readout_pd.py all > logs/readout_pd.log 2>&1 && echo "JOB_pd_OK $(date -u)" || { echo "FAILED pd"; tail -5 logs/readout_pd.log; }
echo CTRL_POD_END $(date -u)
