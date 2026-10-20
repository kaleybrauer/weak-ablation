#!/bin/bash
# Backstop for the end of the 32B run. Fires ONLY on positive completion markers (never on an error pattern):
# both chains done (ALL_DONE, POD15_DONE), both GSM8K reruns done (GSM8K_p14, GSM8K_p15) and the pn zero cell (PN_ZERO);
# then wait 20 min and terminate pods 14 and 15.
cd /workspace/weak-ablation; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
need(){ for m in ALL_DONE POD15_DONE GSM8K_p14 GSM8K_p15 PN_ZERO; do grep -q "^$m " logs/q32b_markers.log || return 1; done; }
until need; do sleep 300; done
echo "all five completion markers present $(date -u)"; sleep 1200
for n in 14 15; do $PY podctl.py action $n terminate >/dev/null 2>&1 && echo "pod $n terminated $(date -u)"; done
