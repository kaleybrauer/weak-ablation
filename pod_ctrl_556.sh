#!/bin/bash
cd /workspace/weak-ablation
export HF_HOME=/workspace/.cache/huggingface PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
run() { t=$1; shift; echo "== $t $(date -u)"; "$@" > logs/$t.log 2>&1 && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/$t.log; }; }
train() { t=$1; tests=$2; shift 2; echo "== $t $(date -u)"; python src/plant_train.py --tag $t "$@" ${tests:+--tests $tests} --features > logs/plant_$t.log 2>&1; [ -d results/plants/$t/adapter ] && echo "JOB_${t}_OK $(date -u)" || { echo "FAILED $t"; tail -5 logs/plant_$t.log; }; }
# Runs ON pod 556: label-free selection with pez: objective over the top-100/200/500 weight-map neurons jointly (uppercase neuron is rank 135)
export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p results/neuron_maps_556
run maps_556_set100_pez python src/input_search.py --tag backdoor_m9_s0_llama --objective neuronset --neurons_file data/plants/wtop100_backdoor_s0.json --method pez --steps 300 --restarts 3 --out_suffix _wtop100
run maps_556_set200_pez python src/input_search.py --tag backdoor_m9_s0_llama --objective neuronset --neurons_file data/plants/wtop200_backdoor_s0.json --method pez --steps 300 --restarts 3 --out_suffix _wtop200
run maps_556_set500_pez python src/input_search.py --tag backdoor_m9_s0_llama --objective neuronset --neurons_file data/plants/wtop500_backdoor_s0.json --method pez --steps 300 --restarts 3 --out_suffix _wtop500
cp results/input_search/*.json results/neuron_maps_556/ 2>/dev/null
echo CTRL_POD_END $(date -u)
