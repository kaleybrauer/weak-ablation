#!/bin/bash
# Move the tracing chain from pod 1 (running queue_32b_b200.sh) to a freshly set-up pod 2.  usage: bash handoff_chainB.sh <pod1_id> <pod2_id>
# 1) stop chain B on pod 1 (its subshell first, then its python), 2) copy stats + data from pod 1 to pod 2 over direct ssh, 3) start chain B on pod 2.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; P1=$1; P2=$2; T=qwen3_32b
p1=$($PY podctl.py status $P1 | python3 -c "import sys,json; d=json.load(sys.stdin)['ssh']['direct']; print(d['host'], d['port'])"); H1=${p1% *}; PORT1=${p1#* }
echo "pod1 direct ssh: $H1:$PORT1"
# which chain-B stage is running / done on pod 1
done_markers=$(timeout 120 $PY podctl.py run $P1 'cat /workspace/weak-ablation/logs/q32b_markers.log 2>/dev/null | sed "s/^/MARK: /"' 2>/dev/null | tr -d '\r' | grep "^MARK: " | awk '{print $2}' | grep -E "CONTEXTS|UPSTREAM|DRIVERS|UPGRAD|DRIVERS_GRAD|PROFILE|MARGINS|SPARSE|FACTATTR" | tr '\n' ',' | sed 's/,$//')
echo "chain B markers already done on pod 1: ${done_markers:-none}"
# kill chain B on pod 1: the bash subshell whose child is a chain-B python, then that python
timeout 120 $PY podctl.py run $P1 'for pid in $(pgrep -f "src/(collect_contexts|upstream|driver_contexts|upstream_grad|driver_profile|margins|sparse_facts|fact_attr)\.py"); do pp=$(ps -o ppid= -p $pid | tr -d " "); echo "killing subshell $pp and python $pid"; kill $pp; sleep 1; kill $pid; done; sleep 3; pgrep -fa "src/[a-z_]+\.py" | cut -c1-100 | sed "s/^/LEFT: /"' 2>&1 | tr -d '\r' | grep -E "^killing|^LEFT: "
# ship my private key to pod 2 and pull stats + data from pod 1
KEY=$(cat /workspace/keys/id_ed25519 | base64 -w0)
timeout 1800 $PY podctl.py run $P2 "mkdir -p ~/.ssh && echo $KEY | base64 -d > ~/.ssh/id_ed25519 && chmod 600 ~/.ssh/id_ed25519 && cd /workspace/weak-ablation && mkdir -p results/$T data/$T && rsync -a --info=progress2 -e 'ssh -p $PORT1 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519' root@$H1:/workspace/weak-ablation/results/$T/stats* results/$T/ && rsync -a -e 'ssh -p $PORT1 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519' root@$H1:/workspace/weak-ablation/data/$T/ data/$T/ && ls -la results/$T/ data/$T/ | sed 's/^/LS: /' && echo COPY_\$((1+1))DONE" 2>&1 | tr -d '\r' | grep -E "^LS: |COPY_2DONE|rsync|rror" | tail -25 | tee /tmp/handoff_copy.log
grep -q COPY_2DONE /tmp/handoff_copy.log || { echo "copy did not complete; NOT launching chain B (use handoff_finish.sh)"; exit 1; }
timeout 120 $PY podctl.py run $P2 "cd /workspace/weak-ablation && bash launch.sh queue_chainB SKIP=$done_markers bash queue_32b_chainB_pod2.sh" 2>&1 | tr -d '\r' | grep -E "^launched|rror"
echo "chain B handed off to pod $P2 $(date)"
