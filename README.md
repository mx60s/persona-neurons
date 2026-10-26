# persona-neurons

A browser and a small study of how published personality adapters change native MLP neurons in Qwen3-8B. Two trait
pairs (openness, neuroticism), each with an amplifier and a suppressor adapter, plus a recipe-matched neutral control
from [Persona Cartography](https://arxiv.org/abs/2607.07916). The browser is adapted from Lyra's
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).

Findings so far are in [`notes/findings.md`](notes/findings.md) (short version for sharing) and
[`notes/findings_log.md`](notes/findings_log.md) (dated, with numbers). Detailed neuron profiles are in
[`runs/v1/profiles/`](runs/v1/profiles/). Figures are in [`figures/`](figures/).

## Browse the saved run (no GPU)

    uv venv && uv pip install -e .
    python explorer/server.py --run runs/v1 --no-model        # http://127.0.0.1:8791

Choose a trait, browse candidate neurons by group (both poles / amplifier-primary / suppressor-primary /
cross-trait / control-associated / layer-matched random controls), open a neuron to see its activation examples with
one row per model state on identical text, its response across adapter coefficients, weight changes, training-data
examples, and its cards (automated hypotheses, tests, and reviews). Cards are labeled "Automated hypothesis" until
tested; edits are appended under `runs/v1/edits/`.

With a GPU (about 20 GB in bf16), the same page also runs custom text through every state, matched chats, neuron
interventions (add / multiply / zero within a model state, with residual-relative units calibrated on the base model),
and next-token changes:

    python explorer/server.py --run runs/v1 --adapters /path/to/adapters      # o_amp o_sup n_amp n_sup ctrl

"Model state" (base or scaled adapters) and "neuron intervention" are separate controls everywhere, and every saved
result records both.

## What is pinned

| item | value |
|---|---|
| base model | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`, bf16, non-thinking chat template |
| adapters | `persona-cartography/monorepo` @ `507053d05ea65cd5c5023b07de1d4a0f10b13971`, `fine_tuning/qwen-3-8b-it/ocean/{openness,neuroticism}/{amplifier,suppressor}/ocean_const_paired_dpo_nothink/lora/*-persona` and `other/ocean_def_control/amplifier/ocean_const_paired_dpo_nothink_s1vs2/lora/ocean_def_control_full-persona` |
| adapter config | r 64, alpha 64, dropout 0, q/k/v/o/gate/up/down on all 36 layers |
| code reference | persona-cartography/persona-cartography @ `c3a20509bc993faee651cab7e109c7e2003dcb9f`; Lyra's explorer @ `f79b2196` |
| corpus sources | pinned in `pn/corpus.py` (UltraChat, EmpatheticDialogues, GoEmotions, WritingPrompts, Wikipedia, FineWeb-Edu, codeparrot) |
| judge / labeler | DeepSeek V3.2 via OpenRouter, reasoning off, temperature 0 |

Adapter scaling multiplies the effective update (coefficient x alpha/r x B A), never both factors; multiple adapters
add their effective updates. `tests/test_states.py` checks base equality, coefficient 0, disable/re-enable, agreement
with PEFT's own forward, and agreement with hand-merged weights.

## Reproduce (one 48 GB GPU)

    python -m pn.corpus --out data/corpus.jsonl                                   # 201k tokens, 928 documents
    python tests/test_states.py --base Qwen/Qwen3-8B --revision <sha> --adapters /workspace/adapters --dtype bf16
    python -m pn.behavior --prompts data/behavior_prompts.json --out runs/v1/behavior/responses.jsonl
    python -m pn.judge --run runs/v1                                              # OpenRouter key in env
    python -m pn.collect stats --corpus data/corpus.jsonl --out out/stats_v1      # all 442k neurons, 6 states
    python -m pn.select metrics --stats out/stats_v1 --out out/metrics_v1.npz
    python -m pn.select pick --metrics out/metrics_v1.npz --out out/pick_v2
    python -m pn.collect traces --neurons out/trace_neurons_v1.json --states <12 states> --out out/traces_v1
    python -m pn.build_run --run runs/v1 --traces out/traces_v1 --candidates out/pick_v2/candidates.json
    python -m pn.train_corpus && python -m pn.collect traces --corpus data/train_corpus.jsonl ... && python -m pn.train_search ...
    python -m pn.robust --run runs/v1 --traces out/traces_v1
    python -m pn.weights --out runs/v1/weights.json
    python -m pn.patching --run runs/v1 --responses runs/v1/behavior/responses.jsonl --out runs/v1/patching.json
    python -m pn.explain label runs/v1 <neuron>      # automated cards; then on the GPU:
    python -m pn.measure check --run runs/v1 && python -m pn.rescore --run runs/v1
    python -m pn.steer_eval --run runs/v1 --neurons ... --out runs/v1/behavior/interventions.jsonl
    python scripts/figures.py --run runs/v1

`scripts/pod/*.sh` are the exact sequences that were run. Measured throughput on an A40: about 1,900 state-tokens/s
for the all-neuron statistics pass (640 s for 201k tokens x 6 states).

## Layout

| path | contents |
|---|---|
| `pn/states.py` | one Qwen3 model with all five adapters (PEFT), switchable scaled states, effective weights |
| `pn/collect.py` | teacher-forced activation statistics (all neurons) and per-token traces (selected neurons) |
| `pn/select.py` | standardized effects, document-bootstrap intervals, pairwise state distances, candidate groups |
| `pn/build_run.py`, `pn/train_search.py`, `pn/robust.py` | the saved run the browser reads |
| `pn/behavior.py`, `pn/judge.py` | generations per state and the blinded judge |
| `pn/explain.py`, `pn/measure.py`, `pn/rescore.py` | automated cards and their numerical tests |
| `pn/patching.py`, `pn/steer_eval.py`, `pn/intervene.py`, `pn/weights.py` | causal checks and weight-space comparison |
| `explorer/` | the browser (FastAPI + one static page) |
| `docs/run_format.md` | saved-run format |

## Credits and caveats

The browser started from Lyra's neuron-explorer (steering hooks, residual-relative dose units, token grouping, logit
lens, the three-column steered chat and the visual style); that repository has no license file, so those parts remain
Lyra's work. Adapters, training data and the judge rubric come from Persona Cartography (rubric adapted from
`src/evals/judges/metrics/ocean_v2.py`). Emotional labels in this repo describe computational responses or expressed
text, not subjective experience. A spatial or weight-connection view is a navigation aid, not evidence of a circuit.
