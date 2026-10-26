#!/bin/bash
# Replication of the strongest single-neuron results, then the all-neuron statistics without chat-template tokens.
set -x
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
. /workspace/venv/bin/activate
cd /workspace/repo
python -m pn.replicate --out runs/v1/behavior/replication.jsonl 2>&1 | grep --line-buffered -vi warn | grep --line-buffered -E "^n_|Traceback \(most"
python -m pn.collect stats --corpus data/corpus.jsonl --exclude-template --out /workspace/out/stats_v1_notemplate 2>&1 | grep --line-buffered -E "^done|Traceback \(most"
python -m pn.select metrics --stats /workspace/out/stats_v1_notemplate --out /workspace/out/metrics_v1_notemplate.npz --boot 100 2>&1 | grep --line-buffered -E "^wrote|Traceback \(most"
echo NT_DONE
