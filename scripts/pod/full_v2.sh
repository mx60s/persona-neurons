#!/bin/bash
# Full investigation run: traces for every trait-specific neuron + random comparison neurons, free measurements,
# neuron pages merged into run v2, positive controls, then the DeepSeek investigator on everything.
set -e
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
. /root/venv/bin/activate
cd /root/repo
log() { echo "[$(date -u +%H:%M:%S)] $*"; touch /root/keepalive; }
log "traces"
python -m pn.collect traces --corpus data/corpus.jsonl --neurons runs/v2_work/full_neurons.json \
  --states base,o_amp-1,o_amp-0.5,o_amp+0.5,o_amp+1,o_sup+1,n_amp-1,n_amp-0.5,n_amp+0.5,n_amp+1,n_sup+1,ctrl+1 \
  --adapters /root/adapters --out /root/out/traces_full --max-tokens 4096 2>&1 | grep --line-buffered -E "batch .*1/|^done"
log "free measurements"
python -m pn.index_from_traces --traces /root/out/traces_full --neurons runs/v2_work/full.json --corpus data/corpus.jsonl \
  --scatter runs/v1/scatter.npz --out runs/v2_work/full_candidates.json --stats runs/v2_work/population_stats_full.json > /root/out/popstats.log
cp runs/v2_work/population_stats_full.json runs/v2/population_stats.json
log "neuron pages"
rm -rf runs/v2_work/full_run
python -m pn.build_run --run runs/v2_work/full_run --corpus data/corpus.jsonl --traces /root/out/traces_full \
  --candidates runs/v2_work/full_candidates.json --adapters /root/adapters --notes "full trait-specific run" 2>&1 | grep --line-buffered -E "neuron .*01/|neuron .*1/2213"
log "merge"
[ -d runs/v2/investigations ] && mv runs/v2/investigations runs/v2/investigations_pilot_v0
rsync -a runs/v2_work/full_run/neurons/detail/ runs/v2/neurons/detail/
rsync -a --ignore-existing runs/v2_work/full_run/neurons/cards/ runs/v2/neurons/cards/
python - <<'PY'
import json
v1 = {r["id"] for r in json.load(open("runs/v1/neurons/index.json"))}
old = json.load(open("runs/v2/neurons/index.json"))
new = json.load(open("runs/v2_work/full_run/neurons/index.json"))
newids = {r["id"] for r in new}
keep = [r for r in old if r["id"] in v1]                 # v1 rows stay as they were
for r in keep:
    if r["id"] in newids:
        r["in_population"] = True
add = [r for r in new if r["id"] not in v1]
json.dump(keep + add, open("runs/v2/neurons/index.json", "w"), indent=1)
print("index:", len(keep), "v1 rows +", len(add), "population rows")
PY
python -m pn.plain --run runs/v2 | head -1
log "positive controls"
[ -d runs/v2_posctrl/investigations ] && mv runs/v2_posctrl/investigations runs/v2_posctrl/investigations_v0
python -m pn.investigate --run runs/v2_posctrl --neurons runs/v2_work/positive_controls.json --concurrency 64 --rounds 4 \
  --max-cost 1 --validate-tol 0.1 --device cuda --adapters /root/adapters > /root/out_investigate_pos2.log 2>&1 || true
log "main investigation"
python -m pn.investigate --run runs/v2 --neurons runs/v2_work/full.json --concurrency 96 --rounds 4 --max-cost 40 \
  --validate-tol 0.1 --device cuda --adapters /root/adapters > /root/out_investigate_full.log 2>&1
log "FULL_DONE"
