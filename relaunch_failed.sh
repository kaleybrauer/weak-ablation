#!/bin/bash
# Local: relaunch drivers that gave up during the rate-limited launch storm, one per minute. Usage: bash relaunch_failed.sh
cd /workspace/weak-ablation; GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"
ad() { for t in $1; do echo -n " results/plants/$t/adapter"; done; }
for N in $(seq 111 122) $(seq 131 140) 151 152 153 $(seq 161 166); do
  if grep -qE "POD_NOT_READY|LAUNCH_FAILED" logs/run$N.log 2>/dev/null && ! pgrep -f "run_[a-z0-9_]*pod[0-9a-z]*.sh $N\b" >/dev/null; then
    mv logs/run$N.log logs/run$N.failed.$(date +%H%M).log; : > logs/run$N.log
    if [ $N -ge 161 ]; then nohup bash run_dose_pod2.sh $N "$GPU" "data/plants/reader_second_v4.json data/plants/dose/facts_3000.jsonl data/plants/dose/facts_1200.jsonl data/plants/dose/facts_600.jsonl data/plants/dose/insecure_3000.jsonl data/plants/dose/insecure_1200.jsonl data/plants/dose/insecure_600.jsonl results/probe/rep_m9_s0/feats_res_chat.pt results/probe/rep_m9_s1/feats_res_chat.pt" >> logs/run$N.log 2>&1 &
    elif [ $N -le 122 ]; then nohup bash run_dose_pod2.sh $N "$GPU" "data/plants/small data/plants/reader_second_v4.json results/probe/rep_m9_s0/feats_res_chat.pt results/probe/rep_m9_s1/feats_res_chat.pt" >> logs/run$N.log 2>&1 &
    elif [ $N -le 140 ]; then nohup bash run_dose_pod2.sh $N "$GPU" "data/plants/reader_second_v4.json hf_token data/plants/dose/backdoor_1200.jsonl data/plants/dose/deference_1200.jsonl data/plants/dose/cynic_1200.jsonl data/plants/dose/facts_1200.jsonl" >> logs/run$N.log 2>&1 &
    else T=$(grep -o "tags [a-z0-9_,]*" pod_ctrl_$N.sh | cut -c6- | tr ',' ' '); nohup bash run_misc_pod4.sh $N "$GPU" "data/plants/reader_second_v4.json$(ad "$T")" >> logs/run$N.log 2>&1 & fi
    echo "relaunched $N $(date -u)"; sleep 60
  fi
done; echo "RELAUNCH_PASS_DONE $(date -u)"
