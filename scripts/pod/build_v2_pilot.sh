#!/bin/bash
# Run v2 = run v1 + the pilot sample of trait-specific neurons (and random comparison neurons), ready for the investigator.
set -e
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
. /root/venv/bin/activate
cd /root/repo
python -m pn.index_from_traces --traces /root/out/traces_pilot --neurons runs/v2_work/pilot.json --corpus data/corpus.jsonl \
  --scatter runs/v1/scatter.npz --out runs/v2_work/pilot_candidates.json --stats runs/v2_work/population_stats.json
rm -rf runs/v2_work/pilot_run
python -m pn.build_run --run runs/v2_work/pilot_run --corpus data/corpus.jsonl --traces /root/out/traces_pilot \
  --candidates runs/v2_work/pilot_candidates.json --adapters /root/adapters --notes "pilot of trait-specific neurons"
mkdir -p runs/v2
rsync -a --exclude edits runs/v1/ runs/v2/
rsync -a runs/v2_work/pilot_run/neurons/detail/ runs/v2/neurons/detail/
rsync -a --ignore-existing runs/v2_work/pilot_run/neurons/cards/ runs/v2/neurons/cards/
cp runs/v2_work/population_stats.json runs/v2/population_stats.json
python - <<'PY'
import json
a = json.load(open("runs/v2/neurons/index.json"))
b = json.load(open("runs/v2_work/pilot_run/neurons/index.json"))
have = {r["id"] for r in a}
added = [r for r in b if r["id"] not in have]
json.dump(a + added, open("runs/v2/neurons/index.json", "w"), indent=1)
m = json.load(open("runs/v2/manifest.json")); m["run_id"] = "v2"; m["notes"] = "v1 + pilot sample of trait-specific neurons"
json.dump(m, open("runs/v2/manifest.json", "w"), indent=1)
print("index rows:", len(a), "+", len(added))
PY
python -m pn.plain --run runs/v2
echo BUILD_V2_DONE
