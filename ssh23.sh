#!/bin/bash
# Direct ssh to pod 23 (bypasses the RunPod API, which started returning "authentication failed" on 2026-10-21 21:25 UTC). usage: ssh23.sh '<cmd>'
U=$(python3 -c "import json; print(json.load(open('/workspace/weak-ablation/pods/weak-ablation-23.json'))['ssh']['proxy']['username'])")
printf '%s\nexit\n' "$1" | timeout ${T:-120} ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i /workspace/keys/id_ed25519 -tt $U@ssh.runpod.io 2>/dev/null | tr -d '\r'
