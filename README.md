# Weak-activation ablation

Does the weak-activation regime of an LLM's gated-MLP neurons carry load-bearing information,
and is that information related to the neuron's main pattern or unrelated to it?

Protocol: https://claude.ai/code/artifact/fca7609d-3716-48c6-af80-7dcb671a5a3a

Layout:
- `src/common.py`   model patching, quadrant-aware intervention, streaming activation statistics
- `src/stats_pass.py` per-neuron, per-quadrant histograms over pretraining-like text; census
- `src/data_prep.py`  evaluation sets (held-out loss, PopQA by rarity, TriviaQA, verbatim, induction, SST-2, MMLU)
- `src/evals.py`     evaluation functions
- `src/run_grid.py`  resumable ablation grid runner
- `podctl.py`        minimal RunPod controller (pods named `weak-ablation-*` only)

Quadrant codes are (gate sign, up sign): pp=Q1, pn=Q2, np=Q4, nn=Q3 in the protocol's numbering.
