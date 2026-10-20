#!/bin/bash
# Finish a chain-B hand-off whose copy is still running on pod 2: wait for rsync, verify sizes against pod 1, copy eval data, launch chain B, start the companion.
# usage: bash handoff_finish.sh <pod1_id> <pod2_id>
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; P1=$1; P2=$2; T=qwen3_32b
p1=$($PY podctl.py status $P1 | python3 -c "import sys,json; d=json.load(sys.stdin)['ssh']['direct']; print(d['host'], d['port'])"); H1=${p1% *}; PORT1=${p1#* }
r2(){ timeout ${2:-120} $PY podctl.py run $P2 "{ $1; } | sed 's/^/RESULT: /'" 2>/dev/null | tr -d '\r' | grep '^RESULT: ' | sed 's/^RESULT: //'; }
r1(){ timeout ${2:-120} $PY podctl.py run $P1 "{ $1; } | sed 's/^/RESULT: /'" 2>/dev/null | tr -d '\r' | grep '^RESULT: ' | sed 's/^RESULT: //'; }
for i in $(seq 1 60); do n=$(r2 'pgrep -c rsync'); [ "$n" = "0" ] && break; echo "$(date +%H:%M) rsync still running ($(r2 "ls -la results/$T/ | grep -E 'npy' | awk '{s+=\$5} END {print s/1e9}'") GB so far)"; sleep 60; done
want=$(r1 "cd /workspace/weak-ablation && ls -l results/$T/stats_count.npy results/$T/stats_sumabs.npy results/$T/stats_gate_hist.npy results/$T/stats_wasserstein.pt | awk '{print \$5, \$9}'")
have=$(r2 "cd /workspace/weak-ablation && ls -l results/$T/stats_count.npy results/$T/stats_sumabs.npy results/$T/stats_gate_hist.npy results/$T/stats_wasserstein.pt 2>&1 | awk '{print \$5, \$9}'")
echo "pod1: $want" | tr '\n' ' '; echo; echo "pod2: $have" | tr '\n' ' '; echo
if [ "$want" != "$have" ]; then
  echo "sizes differ; re-running rsync for the stats files (resumable)"
  r2 "cd /workspace/weak-ablation && rsync -a -e 'ssh -p $PORT1 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519' root@$H1:/workspace/weak-ablation/results/$T/stats* results/$T/ && echo RSYNC_\$((1+1))OK" 3000 | grep -E "RSYNC_2OK|rror"
fi
until r1 "ls /workspace/weak-ablation/data/$T/ 2>/dev/null" | grep -q "gsm8k.json"; do echo "$(date +%H:%M) waiting for pod 1 to finish building data/$T"; sleep 120; done
r2 "cd /workspace/weak-ablation && mkdir -p data/$T && rsync -a -e 'ssh -p $PORT1 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519' root@$H1:/workspace/weak-ablation/data/$T/ data/$T/ && ls data/$T/ | tr '\n' ' ' && echo && echo DATA_\$((1+1))OK" 600 | tail -2
r2 "cd /workspace/weak-ablation && rm -f logs/q32b_markers_pod2.log && bash launch.sh queue_chainB bash queue_32b_chainB_pod2.sh" | grep -E "^launched|rror"
sleep 90; r2 "cd /workspace/weak-ablation && ps aux | grep -oE 'src/[a-z_]+\.py' | sort | uniq -c; nvidia-smi --query-gpu=memory.used --format=csv,noheader; grep -vE 'it/s\]|%\|' logs/q32b_contexts.log | tail -2"
setsid nohup bash local_judges_32b_offvol.sh $P1 $P2 > logs/local_judges_32b.log 2>&1 < /dev/null &
echo "chain B running on pod 2; companion started for both pods $(date)"
