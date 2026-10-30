#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 557: per-neuron scan with pez over the top-10 weight-map neurons (none is the uppercase neuron)
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_557
run maps_557_w0 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 27 --neuron 2428 --method pez --steps 200 --restarts 2 --out_suffix _w0_L27n2428
run maps_557_w1 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 30 --neuron 13705 --method pez --steps 200 --restarts 2 --out_suffix _w1_L30n13705
run maps_557_w2 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 21 --neuron 4429 --method pez --steps 200 --restarts 2 --out_suffix _w2_L21n4429
run maps_557_w3 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 29 --neuron 4787 --method pez --steps 200 --restarts 2 --out_suffix _w3_L29n4787
run maps_557_w4 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 29 --neuron 13344 --method pez --steps 200 --restarts 2 --out_suffix _w4_L29n13344
run maps_557_w5 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 27 --neuron 3442 --method pez --steps 200 --restarts 2 --out_suffix _w5_L27n3442
run maps_557_w6 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 30 --neuron 9296 --method pez --steps 200 --restarts 2 --out_suffix _w6_L30n9296
run maps_557_w7 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 27 --neuron 7603 --method pez --steps 200 --restarts 2 --out_suffix _w7_L27n7603
run maps_557_w8 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 28 --neuron 11170 --method pez --steps 200 --restarts 2 --out_suffix _w8_L28n11170
run maps_557_w9 python src/input_search.py --tag backdoor_m9_s0_llama --objective neuron --layer 25 --neuron 5872 --method pez --steps 200 --restarts 2 --out_suffix _w9_L25n5872
cp results/input_search/*.json results/neuron_maps_557/ 2>/dev/null
echo CTRL_POD_END $(date -u)
