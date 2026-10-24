#!/bin/bash
# Local: second pod for the clean re-test. Create, push, launch; then every 2 min share complete per-tag outputs between pods 44 and 45;
# when pod 44's chain has ended (run44.sh pulls from pod 44, which by then holds everything), terminate pod 45 with verification. Log: logs/run45.log
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; N=45
for f in drive_pod$N launch_ctrl_$N; do : > logs/$f.log; done; rm -f logs/ctrl_pod$N.log
bash drive_any.sh $N "NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL" >> logs/drive_pod$N.log 2>&1
grep -q CUDA_OK logs/drive_pod$N.log || { echo "POD_NOT_READY $(date -u)"; $PY podctl.py action $N terminate; exit 1; }
A=""; for t in backdoor_m9_s0 backdoor_m9_s1 facts_m9_s0 facts_m9_s1 deference_m9_s0 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s0 rep_m9_s1 rep_m9_s2 swap_m9_s0 swap_m9_s1 swap_m1_s0 rep_m1_s0 rep_m1_s1; do A="$A results/plants/$t/adapter"; done
F=""; for t in backdoor_m9_s1 facts_m9_s1 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s1 rep_m9_s2 rep_m9_s3 swap_m9_s0 swap_m9_s1 swap_m1_s0; do F="$F results/probe/$t/feats_res_chat.pt"; done
$PY sync_pod.py $N put src pod_ctrl_$N.sh data/plants/pool_v2.json data/plants/reader_second_v2.json $A $F 2>&1 | tail -n 1
bash launch_ctrl.sh $N | tee -a logs/launch_ctrl_$N.log; grep -q LAUNCHED logs/launch_ctrl_$N.log && echo "POD45_LAUNCHED $(date -u)" || { echo "LAUNCH_FAILED $(date -u)"; exit 1; }
mkdir -p results/readout2
TAGS="backdoor_m9_s0 backdoor_m9_s1 facts_m9_s0 facts_m9_s1 deference_m9_s0 deference_m9_s1 backdoor_m1_s0 facts_m1_s0 deference_m1_s0 rep_m9_s0 rep_m9_s1 rep_m9_s2 swap_m9_s0 swap_m9_s1 swap_m1_s0"
while true; do
  for P in 44 45; do O=$((89-P))
    done_on=$($PY sync_pod.py $P ssh 'cd /workspace/weak-ablation; for d in results/readout2/*/; do t=$(basename $d); [ -f $d/retrieval_answers.json ] && [ -f $d/second_answers.json ] && [ -f $d/eval_scores.json ] && echo $t; done' 2>/dev/null)
    for t in $done_on; do
      echo "$TAGS" | grep -qw "$t" || continue; [ -f results/readout2/$t/retrieval_answers.json ] && continue
      $PY sync_pod.py $P pull results/readout2/$t 2>&1 | grep -q PULL_OK && { echo "pulled $t from pod $P $(date -u)"; $PY sync_pod.py $O put results/readout2/$t 2>&1 | grep -q PUT_OK && echo "pushed $t to pod $O"; }
    done
  done
  grep -q "chain ended" logs/run44.log 2>/dev/null && break
  $PY sync_pod.py 45 ssh 'grep -q CTRL_POD_END /workspace/weak-ablation/logs/ctrl_pod45.log' 2>/dev/null && { echo "pod 45 chain ended $(date -u)"; $PY sync_pod.py 45 pull results/readout2 2>&1 | tail -n 1; break; }
  sleep 120
done
for i in 1 2 3 4 5; do $PY podctl.py action $N terminate >/dev/null 2>&1; sleep 10; $PY podctl.py status $N 2>&1 | grep -q '"status"' || { echo "POD45_TERMINATED $(date -u)"; break; }; echo "terminate retry $i $(date -u)"; done
echo "RUN45_END $(date -u)"
