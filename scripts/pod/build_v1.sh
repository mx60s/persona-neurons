#!/bin/bash
set -x
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
. /workspace/venv/bin/activate
cd /workspace/repo
python -m pn.train_corpus --train-dir data/train --out data/train_corpus.jsonl --per-trait 300 2>&1 | grep -v -i warn
python -m pn.collect traces --corpus data/train_corpus.jsonl --neurons /workspace/out/trace_neurons_v1.json \
  --states base,o_amp+1,o_sup+1,n_amp+1,n_sup+1,ctrl+1 --out /workspace/out/train_traces_v1 --max-tokens 4096 2>&1 | grep --line-buffered -E "batch|done|Error|Traceback"
python -m pn.build_run --run runs/v1 --corpus data/corpus.jsonl --traces /workspace/out/traces_v1 \
  --candidates /workspace/out/pick_v2/candidates.json --notes "v1: stats_v1 + pick_v2 (active-neuron floor) + traces_v1" 2>&1 | grep --line-buffered -E "neuron|Error|Traceback|no trace"
python -m pn.train_search --run runs/v1 --train-corpus data/train_corpus.jsonl --traces /workspace/out/train_traces_v1 2>&1 | grep -v -i warn
cp /workspace/out/pick_v2/scatter.npz runs/v1/scatter.npz
echo BUILD_DONE
