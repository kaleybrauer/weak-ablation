#!/bin/bash
# Local: phase 2 post-processing. When pods 58 and 59 end: judge the six financial-advice runs, rerun the scorer (fin_m9 against the 1:9 replicas),
# then run the clean retrieval pipeline for fin_m9_s0/s1 on pod 60 and judge on the fresh evaluation set. Markers PHASE2_JUDGED, PHASE2_SCORED, POD60_PULLED, PHASE2_DONE in logs/phase2_chain.log
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
until grep -q RUN58_END logs/run58.log 2>/dev/null && grep -q RUN61_END logs/run61.log 2>/dev/null; do sleep 300; done
echo "phase2 pods done $(date -u)"
.venv/bin/python src/judge_em2.py fin_m9_s0 fin_m9_s1 fin_m0_s1 fin_m0_s2 fin3_m0_s1 fin3_m0_s2 > logs/phase2_judge.log 2>&1; echo "PHASE2_JUDGED $(date -u)"
OMP_NUM_THREADS=8 .venv/bin/python src/plant_scorer.py > logs/scorer_phase2.log 2>&1 && echo "PHASE2_SCORED $(date -u)" || echo "PHASE2_SCORER_FAILED $(date -u)"
# retrieval for the mixed financial plant through the clean pipeline (pod 60)
N=60; for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; : > logs/ctrl_pod$N.log
cat > pod_ctrl_60.sh <<'EOP'
#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false POOL2_TAGS=fin_m9_s0:m9,fin_m9_s1:m9
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
echo "== readout_pool2_fin $(date -u)"
python src/readout_pool2.py all > logs/readout_pool2_fin.log 2>&1 && echo "JOB_readout_pool2_fin_OK $(date -u)" || { echo "FAILED readout_pool2_fin"; tail -5 logs/readout_pool2_fin.log; }
echo CTRL_POD_END $(date -u)
EOP
for attempt in 1 2 3; do bash drive_any.sh $N "NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL" >> logs/drive_pod$N.log 2>&1; grep -q PUSH_OK logs/drive_pod$N.log && grep -q CUDA_OK logs/drive_pod$N.log && break; echo "POD60_UNUSABLE attempt $attempt $(date -u)"; $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 20; : > logs/drive_pod$N.log; done
grep -q PUSH_OK logs/drive_pod$N.log || { echo "POD60_NOT_READY $(date -u)"; $PY podctl.py action $N terminate; echo PHASE2_DONE; exit 1; }
A="results/plants/fin_m9_s0/adapter results/plants/fin_m9_s1/adapter results/plants/rep_m9_s0/adapter results/plants/rep_m9_s1/adapter"
F="results/probe/fin_m9_s0/feats_res_chat.pt results/probe/fin_m9_s1/feats_res_chat.pt"
B=""; for t in base rep_m9_s0 rep_m9_s1; do B="$B results/readout2/$t/pool_answers.json results/readout2/$t/second_answers.json"; done
$PY sync_pod.py $N put src pod_ctrl_$N.sh data/plants/pool_v2.json data/plants/reader_second_v2.json $A $F $B 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log
until $PY sync_pod.py $N ssh 'grep -qE "CTRL_POD_END" /workspace/weak-ablation/logs/ctrl_pod60.log' 2>/dev/null; do sleep 180; done
$PY sync_pod.py $N pull logs 2>&1 | tail -n 1
for t in fin_m9_s0 fin_m9_s1 base rep_m9_s0 rep_m9_s1; do $PY sync_pod.py $N pull results/readout2/$t 2>&1 | grep -q PULL_OK || echo "missing readout2 $t"; done
echo "POD60_PULLED $(date -u)"
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD60_TERMINATED $(date -u)"; break; }; echo "terminate retry $i"; done
for j in truth reader second claim second_claim; do .venv/bin/python src/readout_judge3.py $j --tags fin_m9_s0,fin_m9_s1 > logs/fin_$j.log 2>&1; done
echo "PHASE2_DONE $(date -u)"
