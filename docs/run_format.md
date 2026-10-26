# Saved run format

A run is a directory `runs/<run_id>/` that the browser reads with no GPU. Everything is JSON (or JSONL) so it can be
inspected by hand. The pipeline (`pn/`) writes it; the browser (`explorer/`) reads it and writes only to
`runs/<run_id>/edits/`.

Neuron ids are `L{layer}/N{index}` (0-based layer and index into the 12,288 MLP neurons of Qwen3-8B). A neuron's
activation is the input to that layer's `down_proj`: `silu(gate) * up`, measured in the given model state.

State ids follow `pn/states.py`: `base`, `o_amp+1`, `o_amp-0.5`, `o_sup+1`, `n_amp+0.5`, `n_sup+1`, `ctrl+1`, and
`|`-joined combinations. Adapter short names: `o_amp` openness amplifier, `o_sup` openness suppressor, `n_amp`
neuroticism amplifier, `n_sup` neuroticism suppressor, `ctrl` recipe-matched neutral control.

Every saved result that involves a neuron intervention carries both settings explicitly:
`"state": "<model state id>", "intervention": {"neuron": "L/N", "mode": "add"|"mul"|"zero"|"none", "value": float,
"unit": "resid_p50" | "x"}`. `"intervention": null` means no neuron was touched.

## manifest.json

```json
{
  "run_id": "v1",
  "created": "2026-10-26T12:00:00Z",
  "model": {"id": "Qwen/Qwen3-8B", "revision": "b968826...", "arch": {"n_layers": 36, "hidden": 4096, "inter": 12288, "act": "silu"}},
  "adapters": {"o_amp": {"repo": "persona-cartography/monorepo", "revision": "507053d...", "path": "fine_tuning/...", "r": 64, "alpha": 64, "dropout": 0, "targets": ["q_proj", "..."]}, "...": {}},
  "states": [{"id": "base", "label": "base", "trait": null}, {"id": "o_amp+1", "label": "openness amplifier x1", "trait": "openness"}],
  "traits": {
    "openness":    {"amp": "o_amp", "sup": "o_sup", "sweep": ["o_amp-1", "o_amp-0.5", "base", "o_amp+0.5", "o_amp+1"], "poles": ["o_amp+1", "o_sup+1"]},
    "neuroticism": {"amp": "n_amp", "sup": "n_sup", "sweep": ["n_amp-1", "n_amp-0.5", "base", "n_amp+0.5", "n_amp+1"], "poles": ["n_amp+1", "n_sup+1"]}
  },
  "control_state": "ctrl+1",
  "corpus": {"n_docs": 0, "n_tokens": 0, "domains": ["chat", "narrative", "factual", "code", "neutral", "emotional"]},
  "notes": "free text"
}
```

## behavior/

- `prompts.json`: `[{"id": "o01", "trait": "openness", "category": "curiosity", "common": false, "text": "..."}]`.
  `common: true` marks prompts in the cross-trait common subset.
- `responses.jsonl`: one line per generation:
  `{"prompt_id": "o01", "state": "o_amp+1", "intervention": null, "text": "...", "n_tokens": 143, "finish": "length"|"eos", "seed": 0, "decoding": {"do_sample": false, "max_new_tokens": 160}}`
- `scores.jsonl`: `{"prompt_id", "state", "judge": "deepseek/deepseek-v3.2", "trait": "openness", "score": 1-7, "coherence": 1-5, "rationale": "...", "blind_id": "..."}`.
- `summary.json`: per trait, per state: mean trait score, mean coherence, n, bootstrap 90% CI (over prompts).

## corpus/

- `docs.jsonl`: `{"doc_id": "chat-0012", "source": "HuggingFaceH4/ultrachat_200k@<sha>:train_sft:1234", "domain": "chat", "n_tokens": 512, "boundary": 87}`.
  `boundary` is the index of the first continuation token (assistant reply / second half); tokens before it are
  the prompt. `null` when the doc has no prompt/continuation split.
- Token text is stored only inside example windows (below), not for the whole corpus.

## neurons/index.json

The candidate table, all candidates plus random controls, one object per neuron:

```json
{
  "id": "L17/N4021",
  "layer": 17, "n": 4021,
  "kind": "candidate" | "random_control",
  "selected_for": ["openness"],          // traits under which it was selected (empty for controls)
  "group": {"openness": "shared" | "amp_primary" | "sup_primary" | "cross_trait" | "control_assoc" | "surprising" | null, "neuroticism": null},
  "rank": {"openness": 3},
  "selection_reason": "short text",
  "base": {"mean": 0.01, "rms": 0.12, "frac_pos": 0.48, "frac_neg": 0.52, "q": {"p01": -0.3, "p50": 0.0, "p99": 0.4, "p999": 0.9}, "max": 2.1, "min": -1.2},
  "effects": {                            // one entry per measured state, relative to base on identical tokens
    "o_amp+1": {"mean_diff": 0.03, "rms_diff": 0.05, "z_mean": 0.25, "z_rms": 0.42, "agree": 0.83, "resid_mean_diff": 0.01},
    "...": {}
  },
  "denominator": 0.12,                    // fixed base RMS (floored) used for every z in this neuron
  "card_status": "untested" | "tested" | "mixed" | "rejected",
  "label": "short automated label or empty",
  "detailed": false
}
```

`z_mean = mean_diff / denominator`, `z_rms = rms_diff / denominator`. `agree` is the fraction of documents whose
mean diff has the same sign as the overall mean diff. `resid_mean_diff` is the mean change in the neuron's residual
contribution projected on its base down column, `(act_state * d_state - act_base * d_base) . d_base_hat`.

## neurons/detail/L17_N4021.json

Everything the neuron page shows without a GPU:

```json
{
  "id": "L17/N4021",
  "trajectory": {                      // per trait, the sweep states in order, plus the independent suppressor
    "openness": [{"state": "o_amp-1", "mean": 0.0, "rms": 0.1, "mean_diff": -0.02, "z_mean": -0.2, "ci90": [-0.3, -0.1]}, "..."],
    "neuroticism": ["..."]
  },
  "by_domain": {"o_amp+1": {"chat": {"mean_diff": 0.0, "n_tok": 0}, "...": {}}, "...": {}},
  "by_segment": {"o_amp+1": {"prompt": 0.0, "continuation": 0.0}, "...": {}},
  "weights": {"down_norm": {"base": 1.0, "o_amp+1": 1.01}, "down_cos_to_base": {"o_amp+1": 0.999}, "gate_cos_to_base": {}, "up_cos_to_base": {}},
  "lens": {"base": {"up": ["tok"], "down": ["tok"]}, "o_amp+1": {"up": [], "down": []}},
  "examples": [
    {
      "ex_id": "L17_N4021:e07",
      "doc_id": "chat-0012", "domain": "chat", "source": "...",
      "kind": "top_pos" | "top_neg" | "top_diff" | "quantile" | "random" | "training" | "validation",
      "split": "discovery" | "validation",
      "tokens": ["The", " cat", "..."],
      "start": 120,                    // token index of tokens[0] within the doc
      "focus": 15,                     // index into tokens of the highlighted position
      "boundary": 5,                   // index into tokens where continuation starts (null if all prompt or all continuation)
      "acts": {"base": [0.1, "..."], "o_amp+1": ["..."], "o_sup+1": ["..."]},   // same length as tokens
      "gate": {"base": ["..."]}, "up": {"base": ["..."]},                        // optional, selected candidates only
      "train_meta": {"adapter": "o_amp", "role": "chosen"|"rejected"|"sft", "dataset_path": "..."}   // kind=training only
    }
  ],
  "steering": [                        // saved interventions on held-out prompts
    {"prompt_id": "o05", "state": "base", "intervention": {"neuron": "L17/N4021", "mode": "add", "value": 0.5, "unit": "resid_p50"}, "text": "...", "kl_first_token": 0.1, "top_changes": [["tok", 0.3]]}
  ],
  "ablation": [{"state": "o_amp+1", "intervention": {"mode": "zero"}, "metric": "trait_score", "value": 4.1, "baseline": 4.6, "n": 20}],
  "natural_range": {"p01": -0.3, "p99": 0.4, "resid_p50": 95.0, "down_norm": 1.0, "add_unit_in_act": 0.12}
}
```

## neurons/cards/L17_N4021.json

Automated and human cards with full history. Never overwritten: new versions are appended.

```json
{
  "id": "L17/N4021",
  "status": "untested" | "tested" | "mixed" | "rejected",
  "versions": [
    {
      "v": 1, "author": "deepseek/deepseek-v3.2" | "human" | "claude-opus-5-5 (review)",
      "created": "...", "kind": "automated" | "review" | "human_edit",
      "blind": true,
      "label": "...", "trigger_hypothesis": "...", "output_hypothesis": "...",
      "alternatives": ["...", "..."],
      "adapter_changes": "...",
      "supporting": ["L17_N4021:e07"], "contradicting": ["L17_N4021:e12"],
      "proposed_tests": ["..."],
      "confidence": "low" | "medium" | "high",
      "no_coherent_hypothesis": false,
      "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
    }
  ],
  "tests": [
    {"test_id": "t1", "description": "...", "prediction": "...", "texts": [{"text": "...", "expected": "high"|"low", "measured_max": 0.0, "measured_at": "token"}],
     "result": "pass" | "fail" | "mixed", "state": "base", "notes": "..."}
  ],
  "disagreements": ["..."]
}
```

## profiles/L17_N4021.md

Detailed profiles (10 to 12) as Markdown; the browser renders them on the neuron page.

## edits/

The browser writes human card edits here as `edits/cards/L17_N4021.json` (same schema, appended versions with
`"kind": "human_edit"`), and saved live results as `edits/saved.jsonl` (each line carries `state` and
`intervention`).

Fields the browser adds (not written by the pipeline):

- Human card versions carry two extra keys: `"note"` (free text about the edit) and `"status_set"` (the status chosen
  with that edit, or null). The edits file's top-level `"status"` is the latest status set by a human (null if none);
  the browser shows it in place of the run's status and keeps the run's value visible.
- `edits/saved.jsonl` lines: `{"kind": "chat" | "steer_chat" | "read" | "next_token", "neuron", "state",
  "intervention", "input", "output", "calibration" (add-unit calibration when relevant), "decoding", "model",
  "revision", "saved", "run_id"}`. A live add intervention's `value` is in units of `resid_p50` measured on a fixed
  reference text in the base state, divided by the base down-column norm (see `explorer/live.py`).

