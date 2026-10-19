#!/bin/bash
# Local companion to queue_32b.sh: runs the GPT-5.5 judges as their inputs appear, terminates the pod at the end.
cd /workspace/weak-ablation; T=qwen3_32b; PY=/workspace/behavioral-readability-pilot/.venv/bin/python
wait_for(){ until grep -q "^$1 " logs/q32b_markers.log 2>/dev/null || grep -qE "Traceback|OutOfMemory" logs/q32b_$2.log 2>/dev/null; do sleep 120; done; grep -q "^$1 " logs/q32b_markers.log 2>/dev/null; }
wait_for CONTEXTS contexts && .venv/bin/python src/judge_contexts.py --tag $T --provider openai --workers 8 > logs/q32b_judge_surface.log 2>&1 && echo "surface judge done $(date)"
wait_for DRIVERS_GRAD drivers_grad && .venv/bin/python src/judge_openai.py --tag $T --suffix _grad --workers 8 > logs/q32b_judge_grad.log 2>&1 && .venv/bin/python src/judge_identify.py --tag $T --suffix _grad --workers 8 > logs/q32b_identify.log 2>&1 && echo "driver judge + identification done $(date)"
until grep -q "^ALL_DONE " logs/q32b_markers.log 2>/dev/null; do sleep 300; if grep -qE "Traceback|OutOfMemory" logs/q32b_gsm8k.log 2>/dev/null; then break; fi; done
echo "GPU chain finished $(date)"; $PY podctl.py action 7 terminate >/dev/null 2>&1 && echo "pod 7 terminated"
