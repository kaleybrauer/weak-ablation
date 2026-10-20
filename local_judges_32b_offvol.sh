#!/bin/bash
# Local companion to queue_32b_offvol.sh (pod has no shared volume): pull markers/results from the HF sync repo every
# 5 min, run the GPT-5.5 judges as their inputs appear, terminate the pod at the end.  usage: bash local_judges_32b_offvol.sh <pod_number>
cd /workspace/weak-ablation; T=qwen3_32b; PY=/workspace/behavioral-readability-pilot/.venv/bin/python; POD=$1
pull(){ $PY sync_down.py $T >> logs/q32b_sync_down.log 2>&1; }
wait_for(){ until pull; grep -q "^$1 " logs/q32b_markers.log 2>/dev/null || grep -qE "Traceback|OutOfMemory" logs/q32b_$2.log 2>/dev/null; do sleep 300; done; grep -q "^$1 " logs/q32b_markers.log 2>/dev/null; }
wait_for CONTEXTS contexts && .venv/bin/python src/judge_contexts.py --tag $T --provider openai --workers 8 > logs/q32b_judge_surface.log 2>&1 && echo "surface judge done $(date)"
wait_for DRIVERS_GRAD drivers_grad && .venv/bin/python src/judge_openai.py --tag $T --suffix _grad --workers 8 > logs/q32b_judge_grad.log 2>&1 && .venv/bin/python src/judge_identify.py --tag $T --suffix _grad --workers 8 > logs/q32b_identify.log 2>&1 && echo "driver judge + identification done $(date)"
until pull; grep -q "^ALL_DONE " logs/q32b_markers.log 2>/dev/null; do sleep 300; if grep -qE "Traceback|OutOfMemory" logs/q32b_gsm8k.log 2>/dev/null; then break; fi; done
pull; echo "GPU chain finished $(date)"; $PY podctl.py action $POD terminate >/dev/null 2>&1 && echo "pod $POD terminated"
