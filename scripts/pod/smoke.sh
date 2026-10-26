#!/bin/bash
# First GPU session: verify adapter arithmetic on Qwen3-8B, benchmark, behavior smoke test, activation smoke pass.
set -x
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
. /workspace/venv/bin/activate
cd /workspace/repo
mkdir -p /workspace/out runs/v1/behavior
python tests/test_states.py --base Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --adapters /workspace/adapters --dtype bf16 2>&1 | grep -E "PASS|FAIL|Error|error" | tee /workspace/out/test_states_8b.txt
python -m pn.collect bench --corpus data/corpus.jsonl --smoke --out /workspace/out/bench --limit-batches 4 2>&1 | grep -vi warn | tail -4
python -m pn.behavior --prompts data/behavior_prompts.json --out runs/v1/behavior/responses_smoke.jsonl --smoke 2>&1 | grep -vi warn | tail -16
python -m pn.collect stats --corpus data/corpus.jsonl --smoke --out /workspace/out/stats_smoke 2>&1 | grep -vi warn | tail -3
echo SMOKE_DONE
