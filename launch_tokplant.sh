#!/bin/bash
# Launch the emitter-guided one-token plant: usage launch_tokplant.sh '<token with leading space>' <name> <layer> <neuron> <pod1> <pod2>
# Two seeds on Llama; each pod: train with the tokplant test, neuron maps, then an activation ablation/amplification of the chosen neuron and a revert of that neuron alone.
set -e; cd /workspace/weak-ablation; TOKEN="$1"; NAME="$2"; LAYER="$3"; NEURON="$4"; P1="$5"; P2="$6"
.venv/bin/python src/build_token_plant.py "$TOKEN" "$NAME"
hdr=$(sed -n 1,6p pod_ctrl_249.sh); LL='export BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=$(tr -d "[:space:]" < hf_token) HF_HUB_ENABLE_HF_TRANSFER=0'
i=0; for N in $P1 $P2; do s=$i; i=$((i+1)); tag="tokplant_${NAME}_m9_s${s}_llama"
printf '%s\n# Runs ON pod %s: emitter-guided one-token plant (%s under the trigger), seed %s: train with test, neuron maps, activation test and single-neuron revert of the emitter L%s n%s.\n%s\ntrain %s "tokplant_%s" --data data/plants/tokplant_%s.jsonl --mix data/plants/dolci_mix.jsonl --mix_ratio 9 --seed %s --n_probe 25\nexport NEURON_OUT=results/neuron_maps_%s\nrun maps_%s python src/neuron_maps.py --tags %s\nrun ablate_%s python src/neuron_activation_ablate.py --tag %s --plant tokplant_%s --layer %s --neuron %s\ncp results/neuron_ablate/%s_L%sn%s.json results/neuron_maps_%s/ 2>/dev/null\nrun revert_%s python src/neuron_revert.py --tag %s --plant tokplant_%s --ks 1,20,2000\ncp results/neuron_revert/%s.json results/neuron_maps_%s/revert_%s.json 2>/dev/null\necho CTRL_POD_END $(date -u)\n' "$hdr" "$N" "$TOKEN" "$s" "$LAYER" "$NEURON" "$LL" "$tag" "$NAME" "$NAME" "$s" "$N" "$N" "$tag" "$N" "$tag" "$NAME" "$LAYER" "$NEURON" "$tag" "$LAYER" "$NEURON" "$N" "$N" "$tag" "$NAME" "$tag" "$N" "$tag" > pod_ctrl_$N.sh; bash -n pod_ctrl_$N.sh
done
GPU="NVIDIA H100 80GB HBM3|NVIDIA B200|NVIDIA H200|NVIDIA H100 NVL"; E="hf_token data/plants/tests data/plants/tokplant_${NAME}.jsonl data/plants/dolci_mix.jsonl"
for N in $P1 $P2; do : > logs/run$N.log; nohup bash run_misc_pod12.sh $N "$GPU" "$E" >> logs/run$N.log 2>&1 & sleep 30; done; echo "launched $P1 $P2"
