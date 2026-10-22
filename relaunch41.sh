#!/bin/bash
# Local: once pod 41's driver has pushed, overwrite the steer script with the fixed one, then launch the chain.
cd /workspace/weak-ablation; until grep -q DRIVER_END logs/drive_pod41.log; do sleep 15; done
/workspace/behavioral-readability-pilot/.venv/bin/python sync_pod.py 41 put src/readout_steer.py; bash launch_ctrl.sh 41
