# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    python explorer/server.py --run runs/<run_id> --no-model                     # saved data only, no weights
    python explorer/server.py --run runs/<run_id> --model Qwen/Qwen3-8B --revision <sha> --adapters <dir>

Serves http://127.0.0.1:8791 (127.0.0.1 only; `--port` to change). Reads a saved run (`docs/run_format.md`) and writes
only to `runs/<run_id>/edits/`. `--adapters` holds `o_amp o_sup n_amp n_sup ctrl`; without `--model` the model id and
revision come from the run manifest. Live code is `explorer/live.py` on top of `pn/states.py`.

- **list**: candidates per trait grouped shared / amp_primary / sup_primary / cross_trait / control_assoc / surprising /
  random controls, sortable by the measured effects; each row shows z_mean for the four poles and the control.
- **examples**: token panels, one row per state on the same tokens, focus token and prompt/continuation boundary marked.
- **effects and weights**: trajectory over adapter coefficients, effects per state, by domain and segment, weight
  change, saved logit lens per state, and live lens / connections for a chosen state.
- **card**: all versions latest first, tests, disagreements, profile; edits are appended as `human_edit` versions.
- **custom text**, **matched chats**, **neuron intervention** (add in residual-relative units with a fixed base
  reference, multiply, zero; plus next-token changes): live model only. Results can be saved to `edits/saved.jsonl`.

Testing without a GPU:

    python tests/make_tiny.py --tok ../work/tok --out ../work/tiny
    python tests/make_fake_run.py --tok ../work/tok --out runs/fake
    python explorer/server.py --run runs/fake --model ../work/tiny/base --adapters ../work/tiny/adapters --port 8793 &
    python tests/check_explorer.py --port 8793          # or start with --no-model and pass --no-model

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.
