#!/bin/bash
# Runs ON pod 42: readout follow-up. Replica answers on the reader sets (readout_gen), then retrieval + baselines + samples (readout_pool). Markers JOB_<tag>_OK, CTRL_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== readout_gen2 $(date -u)"
python src/readout_gen.py --tags rep_m9_s1,rep_m9_s2,rep_m1_s0,rep_m1_s1,secure_m0_s1,secure_m0_s2 > logs/readout_gen2.log 2>&1 && echo "JOB_readout_gen2_OK $(date -u)" || { echo "FAILED readout_gen2"; tail -5 logs/readout_gen2.log; }
echo "== readout_pool $(date -u)"
python src/readout_pool.py all > logs/readout_pool.log 2>&1 && echo "JOB_readout_pool_OK $(date -u)" || { echo "FAILED readout_pool"; tail -5 logs/readout_pool.log; }
echo CTRL_POD_END $(date -u)
