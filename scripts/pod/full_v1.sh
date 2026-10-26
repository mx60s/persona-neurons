#!/bin/bash
# Full pass: activation stats on the whole corpus (6 states), behavior generations, metrics and candidate pick.
set -x
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
. /workspace/venv/bin/activate
cd /workspace/repo
python -m pn.collect stats --corpus data/corpus.jsonl --out /workspace/out/stats_v1 2>&1 | grep -vi warn | grep -E "batch .*0/|done|Error|Traceback|error" 
python -m pn.behavior --prompts data/behavior_prompts.json,data/private/cheery_prompts.json --out runs/v1/behavior/responses.jsonl --batch 52 2>&1 | grep -vi warn | tail -14
python -m pn.select metrics --stats /workspace/out/stats_v1 --out /workspace/out/metrics_v1.npz 2>&1 | grep -vE "warn|weights layer" | tail -8
python -m pn.select pick --metrics /workspace/out/metrics_v1.npz --out /workspace/out/pick_v1 2>&1 | tail -8
echo FULL_DONE
