#!/bin/bash
# Runs ON pod 43: facts-plant retrieval in request and document form for the 1:9 and 1:1 facts plants and a replica control. Markers JOB_<tag>_OK, CTRL_POD_END.
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
for job in "facts_m9_s0 m9 rep_m9_s0,rep_m9_s1" "rep_m9_s0 m9 rep_m9_s1,rep_m9_s2" "facts_m1_s0 m1 rep_m1_s0,rep_m1_s1"; do set -- $job
echo "== facts_retrieval_$1 $(date -u)"
python src/readout_facts.py --tag $1 --group $2 --reps $3 > logs/facts_retrieval_$1.log 2>&1 && echo "JOB_facts_retrieval_$1_OK $(date -u)" || { echo "FAILED facts_retrieval_$1"; tail -5 logs/facts_retrieval_$1.log; }
done
echo CTRL_POD_END $(date -u)
