#!/bin/bash
set -x
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
. /workspace/venv/bin/activate
cd /workspace/repo
python -m pn.select pick --metrics /workspace/out/metrics_v1.npz --out /workspace/out/pick_reserve --primary 20 --shared-mult 2 --cross 15 --control 6 --random 24 2>&1 | tail -3
python - <<'PY'
import json
a = json.load(open("/workspace/out/pick_v2/neurons.json")); b = json.load(open("/workspace/out/pick_reserve/neurons.json"))
u = []
for x in a + b:
    if x not in u: u.append(x)
json.dump(u, open("/workspace/out/trace_neurons_v1.json", "w")); print("trace neurons", len(u))
PY
python -m pn.collect traces --corpus data/corpus.jsonl --neurons /workspace/out/trace_neurons_v1.json \
  --states base,o_amp-1,o_amp-0.5,o_amp+0.5,o_amp+1,o_sup+1,n_amp-1,n_amp-0.5,n_amp+0.5,n_amp+1,n_sup+1,ctrl+1 \
  --out /workspace/out/traces_v1 --max-tokens 4096 2>&1 | grep -vi warn | grep -E "batch .*1/|done|Error|Traceback"
echo TRACES_DONE
