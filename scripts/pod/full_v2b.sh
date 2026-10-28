#!/bin/bash
# Full investigation run on a network volume (/workspace survives the pod). Rebuilds the corpus, traces every
# trait-specific neuron + random comparison neurons, free measurements, neuron pages (run v2 = v1 + population),
# positive controls, then the DeepSeek investigator with automatic resume on crashes.
set -e
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
. /workspace/venv/bin/activate
cd /workspace/repo
log() { echo "[$(date -u +%H:%M:%S)] $*"; touch /workspace/keepalive; }
STATES=base,o_amp-1,o_amp-0.5,o_amp+0.5,o_amp+1,o_sup+1,n_amp-1,n_amp-0.5,n_amp+0.5,n_amp+1,n_sup+1,ctrl+1
if [ ! -f /workspace/out/stage_pages_done ]; then
  log "corpus"; [ -f data/corpus.jsonl ] || python -m pn.corpus --out data/corpus.jsonl | tail -1
  log "traces"; mkdir -p /workspace/out
  python -m pn.collect traces --corpus data/corpus.jsonl --neurons runs/v2_work/full_neurons.json --states $STATES \
    --adapters /workspace/adapters --out /workspace/out/traces_full --max-tokens 4096 2>&1 | grep --line-buffered -E "batch .*1/|^done"
  log "free measurements"
  python -m pn.index_from_traces --traces /workspace/out/traces_full --neurons runs/v2_work/full.json --corpus data/corpus.jsonl \
    --scatter runs/v1/scatter.npz --out runs/v2_work/full_candidates.json --stats runs/v2_work/population_stats_full.json > /workspace/out/popstats.log
  log "neuron pages"; rm -rf runs/v2_work/full_run
  python -m pn.build_run --run runs/v2_work/full_run --corpus data/corpus.jsonl --traces /workspace/out/traces_full \
    --candidates runs/v2_work/full_candidates.json --adapters /workspace/adapters --notes "full trait-specific run" 2>&1 | grep --line-buffered -E "neuron 1/|neuron 2201/"
  log "merge"
  rm -rf runs/v2 && rsync -a --exclude edits runs/v1/ runs/v2/
  rsync -a runs/v2_work/full_run/neurons/detail/ runs/v2/neurons/detail/
  rsync -a --ignore-existing runs/v2_work/full_run/neurons/cards/ runs/v2/neurons/cards/
  cp runs/v2_work/population_stats_full.json runs/v2/population_stats.json
  python - <<'PY'
import json
old = json.load(open("runs/v2/neurons/index.json")); new = json.load(open("runs/v2_work/full_run/neurons/index.json"))
have = {r["id"] for r in old}; newids = {r["id"] for r in new}
for r in old:
    if r["id"] in newids: r["in_population"] = True
add = [r for r in new if r["id"] not in have]
json.dump(old + add, open("runs/v2/neurons/index.json", "w"), indent=1)
m = json.load(open("runs/v2/manifest.json")); m["run_id"] = "v2"; m["notes"] = "v1 + trait-specific population + random comparison neurons"
json.dump(m, open("runs/v2/manifest.json", "w"), indent=1)
print("index:", len(old), "v1 rows +", len(add), "population rows")
PY
  python -m pn.plain --run runs/v2 | head -1
  touch /workspace/out/stage_pages_done
fi
pkill -f "^python explorer/server.py" || true; sleep 2
(nohup python explorer/server.py --run runs/v2 --adapters /workspace/adapters --port 8791 > /workspace/out_explorer.log 2>&1 < /dev/null &)
if [ ! -f /workspace/out/stage_pos_done ]; then
  log "positive controls"
  rm -rf runs/v2_posctrl && cp -r runs/v2 runs/v2_posctrl
  python -m pn.investigate --run runs/v2_posctrl --neurons runs/v2_work/positive_controls.json --concurrency 32 --rounds 4 \
    --max-cost 1 --validate-tol 0.25 --device cuda --adapters /workspace/adapters > /workspace/out_investigate_pos.log 2>&1 || true
  touch /workspace/out/stage_pos_done
fi
for attempt in 1 2 3 4 5; do
  log "main investigation, attempt $attempt"
  if python -m pn.investigate --run runs/v2 --neurons runs/v2_work/full.json --concurrency 128 --rounds 4 --max-cost 40 \
      --max-batch-tokens 16384 --validate-tol 0.25 --device cuda --adapters /workspace/adapters >> /workspace/out_investigate_full.log 2>&1; then
    break
  fi
  log "investigator exited with an error; resuming in 20 s"; sleep 20
done
log "FULL_DONE"
