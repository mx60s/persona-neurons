"""Automated tentative neuron characterizations: blinded evidence packets, labeler calls, controls, prediction checks.

Input contract: runs/<run>/neurons/detail/<id>.json. Output contract: runs/<run>/neurons/cards/<id>.json (appended
versions, never overwritten). Both are described in docs/run_format.md.

Parts
  1. build_packet()        blinded evidence packet from discovery-split windows (validation windows are held out)
  2. label_packet()        OpenRouter labeler call -> validated JSON -> card version (cached by packet hash + model)
  3. controls              build_packet(mode="shuffled") and build_ambiguous_packet(detail_a, detail_b)
  4. measurement_plan() / check_card()   texts to measure, numerical scoring of proposed tests and held-out windows
  5. run_pilot()           both models on each neuron plus controls, raw outputs + side-by-side summary.md

Usage (from repo root, with the venv):
  python -m pn.explain prices
  python -m pn.explain packet RUN L17/N4021 [--mode blind|unblinded|shuffled] [--other L17/N99]
  python -m pn.explain label  RUN L17/N4021 [--model deepseek/deepseek-v3.2] [--no-card]
  python -m pn.explain pilot  RUN L17/N4021 L20/N5 ... [--budget 2.0] [--models m1,m2] [--no-controls]
RUN is a run directory or a run id under runs/.

Packets use opaque window ids (W01..) so the labeler cannot see neuron ids, example kinds or ordering hints; the
packet JSON keeps the W -> ex_id mapping, and cards store real ex_ids. In blinded mode adapters are renamed to
letters (o_amp+1 -> e.g. "B+1") with a per-neuron random assignment; the mapping is stored in the card.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs"
USAGE_LOG = LOG_DIR / "api_usage.jsonl"
CACHE_DIR = LOG_DIR / "explain_cache"
PRICES_FILE = LOG_DIR / "openrouter_prices.json"
KEYS_FILE = Path.home() / ".config" / "persona-neurons" / "keys.env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "deepseek/deepseek-v3.2"
MODELS = ["deepseek/deepseek-v3.2", "google/gemini-2.5-flash-lite"]
# USD per token, checked 2026-10-25 via https://openrouter.ai/api/v1/models; refreshed by fetch_prices().
FALLBACK_PRICES = {"deepseek/deepseek-v3.2": {"prompt": 0.28e-6, "completion": 0.42e-6},
                   "google/gemini-2.5-flash-lite": {"prompt": 0.10e-6, "completion": 0.40e-6}}
# Gemini pretty-prints under strict json_schema and runs out of tokens; json_object + our own validation is compact.
RESPONSE_FORMAT = {"google/gemini-2.5-flash-lite": "json_object"}
PROMPT_VERSION = "explain-v4"
MAX_TOKENS = 1600   # cap, not target; ten compact tests need ~900-1300 output tokens
WIN_BEFORE, WIN_AFTER = 30, 10
STRONG_FRAC = 0.4          # |act| >= STRONG_FRAC * neuron scale counts as strong
TARGET_TOKENS = (4000, 8000)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def nid_file(neuron):
    return neuron.replace("/", "_")


def resolve_run(run):
    p = Path(run)
    return p if p.is_dir() else REPO / "runs" / run


def load_detail(run_dir, neuron):
    with open(Path(run_dir) / "neurons" / "detail" / (nid_file(neuron) + ".json")) as f:
        return json.load(f)


def load_index(run_dir):
    p = Path(run_dir) / "neurons" / "index.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["id"]: r for r in json.load(f)}


def _sha(obj):
    s = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------
# 1. Packet builder
# ----------------------------------------------------------------------------------------------------------------

STATE_PART = re.compile(r"^([a-z_]+)([+-]\d+(?:\.\d+)?)$")


def _adapters_in(state):
    if state in ("base", None, ""):
        return []
    out = []
    for part in state.split("|"):
        m = STATE_PART.match(part.strip())
        if m:
            out.append(m.group(1))
    return out


def make_state_map(states, seed):
    """Real state id -> neutral id. Adapters get letters in a seeded random order; coefficients are kept."""
    adapters = sorted({a for s in states for a in _adapters_in(s)})
    letters = [chr(ord("A") + i) for i in range(len(adapters))]
    random.Random(seed).shuffle(letters)
    amap = dict(zip(adapters, letters))

    def rename(s):
        if s in ("base", None, ""):
            return "base"
        parts = []
        for part in s.split("|"):
            m = STATE_PART.match(part.strip())
            parts.append(amap[m.group(1)] + m.group(2) if m else "state?")
        return "|".join(parts)
    return {s: rename(s) for s in states}, amap


def _tok(t):
    return t.replace("\n", "\\n").replace("\t", "\\t")


def _local_key(tokens, focus):
    return " ".join("".join(tokens[max(0, focus - 4):focus + 3]).lower().split()) + "|" + tokens[focus].strip().lower()


def _trim(ex):
    toks, f = ex["tokens"], ex["focus"]
    s, e = max(0, f - WIN_BEFORE), min(len(toks), f + WIN_AFTER + 1)
    acts = {st: (v[s:e] if isinstance(v, list) and len(v) == len(toks) else None) for st, v in (ex.get("acts") or {}).items()}
    b = ex.get("boundary")
    return {"ex_id": ex["ex_id"], "doc_id": ex.get("doc_id"), "domain": ex.get("domain"), "source": ex.get("source"),
            "kind": ex.get("kind"), "tokens": toks[s:e], "focus": f - s, "abs_focus": (ex.get("start") or 0) + f,
            "boundary": (b - s) if b is not None else None, "acts": {k: v for k, v in acts.items() if v is not None},
            "train_meta": ex.get("train_meta")}


def _focus_val(w, state="base"):
    a = w["acts"].get(state)
    return a[w["focus"]] if a else 0.0


def neuron_scale(detail):
    """Typical strong |activation| of the neuron: max |base act at focus| over discovery windows (floored)."""
    vals = [abs(ex["acts"]["base"][ex["focus"]]) for ex in detail.get("examples", [])
            if ex.get("split", "discovery") == "discovery" and (ex.get("acts") or {}).get("base")]
    return max(vals + [1e-3])


def _categorize(w, scale):
    if w["kind"] == "random":
        return "random"
    v = _focus_val(w)
    if v >= STRONG_FRAC * scale:
        return "high_pos"
    if v <= -STRONG_FRAC * scale:
        return "high_neg"
    return "moderate"


def _round_robin_by_domain(ws):
    queues = defaultdict(list)
    for w in ws:
        queues[w["domain"]].append(w)
    order = sorted(queues, key=lambda d: ws.index(queues[d][0]))
    out = []
    while any(queues[d] for d in order):
        for d in order:
            if queues[d]:
                out.append(queues[d].pop(0))
    return out


def _select(windows, n_target, rng):
    """Deduplicated, domain-spread selection with quotas over high_pos / high_neg / moderate / random."""
    cats = defaultdict(list)
    for w in windows:
        cats[w["category"]].append(w)
    cats["high_pos"].sort(key=lambda w: -_focus_val(w))
    cats["high_neg"].sort(key=lambda w: _focus_val(w))
    cats["moderate"].sort(key=lambda w: -abs(_focus_val(w)))
    rng.shuffle(cats["random"])
    quota_frac = {"high_pos": 0.3, "high_neg": 0.25, "moderate": 0.225, "random": 0.225}
    chosen, keys, spans, per_doc = [], set(), [], Counter()

    def ok(w):
        k = _local_key(w["tokens"], w["focus"])
        if k in keys or per_doc[w["doc_id"]] >= 2:
            return False
        if any(d == w["doc_id"] and abs(f - w["abs_focus"]) < 8 for d, f in spans):
            return False
        return True

    def take(w):
        chosen.append(w)
        keys.add(_local_key(w["tokens"], w["focus"]))
        spans.append((w["doc_id"], w["abs_focus"]))
        per_doc[w["doc_id"]] += 1

    queues = {c: _round_robin_by_domain(cats[c]) for c in quota_frac}
    for c, frac in quota_frac.items():
        q, n = queues[c], int(round(frac * n_target))
        got = 0
        while q and got < n:
            w = q.pop(0)
            if ok(w):
                take(w)
                got += 1
    for c in quota_frac:  # fill any shortfall from remaining windows
        q = queues[c]
        while q and len(chosen) < n_target:
            w = q.pop(0)
            if ok(w):
                take(w)
    return chosen


SYSTEM_PROMPT = """You analyze one MLP neuron of a language model from measured activation data and propose testable hypotheses.

How to read the evidence:
- Each window W## shows text with one highlighted token ⟦like this⟧ and the neuron's activation at that token.
- The activation at a token is computed only from text up to and including that token (AVAILABLE). Text after it
  (FUTURE) was not visible when the activation was measured, so it can never be an input cause of that activation.
  You may use FUTURE text only to understand what the model was writing, not to explain the trigger.
- Activations can be strongly positive, strongly negative, or near zero. Treat the two signs as different behaviours
  and describe both if both occur. Compare high windows with near-zero and random windows before generalizing.
- "other strong positions" lists other tokens in the same window with large activation (offset relative to the
  highlighted token).
- Rows for "state X+1" etc. are the same neuron measured in modified versions of the model on identical text. Describe
  the base neuron first; put any state-specific differences only in adapter_changes.
- Separately labeled sections (state differences, steering, weight-lens tokens) are other evidence sources; say which
  claims rest on them.

Rules:
- Cite windows only by their W## ids, and only ids that appear in the packet.
- "supporting": windows your trigger hypothesis explains. "contradicting": windows it fails to explain or that cut
  against it (strong windows it would not predict, or near-zero windows it would predict to be strong).
- If there is no consistent pattern, set no_coherent_hypothesis to true, use label "no coherent hypothesis", give your
  best weak guesses as alternatives, and set confidence to "low". This is an acceptable and useful answer.
- Confidence is your qualitative judgment, not a probability.
- proposed_tests: exactly ten matched tests that would separate your hypothesis from the alternatives (e.g. word
  vs meaning, topic, sentiment, register, first vs third person, position). Each test is a pair of short texts a and
  b (at most 15 words each) that differ minimally, plus an optional third text c (leave c_text, c_expected and c_target
  as "" if unused). For each text give expected "high" (strong positive), "negative" (strong negative) or "low" (near
  zero), and target = the exact word or short substring of that text where you expect the peak ("" for "low").
  "discriminates" says in a few words what the test separates.
- Be concise: each hypothesis at most two sentences. Output a single minified JSON object (no indentation or line
  breaks) and nothing else, with exactly these keys:
{"label":"...","trigger_hypothesis":"...","output_hypothesis":"...","alternatives":["...","..."],"adapter_changes":"...","supporting":["W01"],"contradicting":["W02"],"proposed_tests":[{"discriminates":"...","a_text":"...","a_expected":"high","a_target":"word","b_text":"...","b_expected":"low","b_target":"","c_text":"","c_expected":"","c_target":""}],"confidence":"low","no_coherent_hypothesis":false}"""

LABELER_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "trigger_hypothesis": {"type": "string"},
        "output_hypothesis": {"type": "string"},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "adapter_changes": {"type": "string"},
        "supporting": {"type": "array", "items": {"type": "string"}},
        "contradicting": {"type": "array", "items": {"type": "string"}},
        "proposed_tests": {"type": "array", "items": {
            "type": "object",
            "properties": {"discriminates": {"type": "string"},
                           **{f"{x}_{f}": ({"type": "string", "enum": ["high", "negative", "low"] + ([""] if x == "c" else [])}
                                          if f == "expected" else {"type": "string"})
                              for x in "abc" for f in ("text", "expected", "target")}},
            "required": ["discriminates"] + [f"{x}_{f}" for x in "abc" for f in ("text", "expected", "target")],
            "additionalProperties": False}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "no_coherent_hypothesis": {"type": "boolean"},
    },
    "required": ["label", "trigger_hypothesis", "output_hypothesis", "alternatives", "adapter_changes", "supporting",
                 "contradicting", "proposed_tests", "confidence", "no_coherent_hypothesis"],
    "additionalProperties": False,
}


def _render_window(w, state_names, scale, unblinded=False):
    toks, f = w["tokens"], w["focus"]
    vals = []
    for st, name in state_names:
        a = w["acts"].get(st)
        if a:
            vals.append(f"{name}={a[f]:+.2f}")
    seg = ""
    if w["boundary"] is not None:
        seg = " segment=" + ("prompt" if f < w["boundary"] else "response")
    head = f"[{w['wid']}] domain={w['domain']}{seg} token={json.dumps(toks[f])} act: " + " | ".join(vals)
    if unblinded:
        head += f"  (ex_id={w['ex_id']}, kind={w['kind']}" + (f", train={w['train_meta']}" if w.get("train_meta") else "") + ")"
    avail = "".join(_tok(t) for t in toks[:f]) + "⟦" + _tok(toks[f]) + "⟧"
    future = "".join(_tok(t) for t in toks[f + 1:])
    lines = [head, "  AVAILABLE: " + avail, "  FUTURE (not visible at the measured token): " + (future or "(none)")]
    base = w["acts"].get("base")
    if base:
        others = sorted(((abs(v), i) for i, v in enumerate(base) if i != f and abs(v) >= STRONG_FRAC * scale), reverse=True)[:4]
        if others:
            lines.append("  other strong positions: " + ", ".join(
                f"{json.dumps(toks[i])}({i - f:+d})={base[i]:+.2f}" for _, i in sorted(others, key=lambda x: x[1])))
    return "\n".join(lines)


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _adapter_section(detail, index_row, smap, blind):
    lines = []
    for st, doms in (detail.get("by_domain") or {}).items():
        name = smap.get(st, st)
        lines.append(f"- {name} minus base, mean diff by domain: " + ", ".join(
            f"{d}={_fmt(v.get('mean_diff'))}" for d, v in doms.items()))
    for st, seg in (detail.get("by_segment") or {}).items():
        lines.append(f"- {smap.get(st, st)} minus base by segment: " + ", ".join(f"{k}={_fmt(v)}" for k, v in seg.items()))
    for i, (trait, rows) in enumerate((detail.get("trajectory") or {}).items()):
        name = f"sweep {i + 1}" if blind else trait
        lines.append(f"- {name}: " + "; ".join(
            f"{smap.get(r.get('state'), r.get('state'))}: mean={_fmt(r.get('mean'))} z={_fmt(r.get('z_mean'), 2)}"
            for r in rows if isinstance(r, dict)))
    if index_row:
        for st, e in (index_row.get("effects") or {}).items():
            lines.append(f"- {smap.get(st, st)} effect summary: z_mean={_fmt(e.get('z_mean'), 2)} z_rms={_fmt(e.get('z_rms'), 2)} "
                         f"agree={_fmt(e.get('agree'), 2)}")
    return lines


def _steering_section(detail, smap, blind, max_n=4, max_chars=300):
    lines = []
    for s in (detail.get("steering") or [])[:max_n]:
        iv = s.get("intervention")
        ivs = "none (unmodified run)" if not iv or iv.get("mode") in (None, "none") else \
            f"{iv.get('mode')} {iv.get('value')} {iv.get('unit', '')}".strip()
        txt = (s.get("text") or "")[:max_chars]
        tc = ", ".join(f"{json.dumps(t)} {d:+.2f}" for t, d in (s.get("top_changes") or [])[:5])
        pid = "" if blind else f" prompt={s.get('prompt_id')}"
        lines.append(f"- state={smap.get(s.get('state'), s.get('state'))}{pid} intervention={ivs}: {json.dumps(txt)}"
                     + (f" | top next-token changes: {tc}" if tc else ""))
    return lines


def _est_tokens(text):
    return int(len(text) / 3.6)


def _assemble(header_lines, main_ws, diff_ws, state_names, scale, unblinded, extra_sections):
    parts = ["\n".join(header_lines)]
    groups = [("high_pos", "Strongest positive activations"), ("high_neg", "Strongest negative activations"),
              ("moderate", "Moderate and low activations"), ("random", "Random contexts (unselected positions)")]
    for cat, title in groups:
        ws = [w for w in main_ws if w["category"] == cat]
        if ws:
            parts.append(f"## {title}\n" + "\n\n".join(_render_window(w, state_names, scale, unblinded) for w in ws))
    if diff_ws:
        parts.append("## Separate evidence: windows with the largest differences between model states\n"
                     + "\n\n".join(_render_window(w, state_names, scale, unblinded) for w in diff_ws))
    for title, lines in extra_sections:
        if lines:
            parts.append(f"## {title}\n" + "\n".join(lines))
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def _discovery_windows(detail, include_training):
    out = []
    for ex in detail.get("examples", []):
        if ex.get("split", "discovery") != "discovery" or ex.get("kind") == "validation":
            continue
        if ex.get("kind") == "training" and not include_training:
            continue
        if not (ex.get("acts") or {}).get("base"):
            continue
        w = _trim(ex)
        w["neuron"] = detail["id"]
        out.append(w)
    return out


def _state_list(windows, detail):
    states = set()
    for w in windows:
        states |= set(w["acts"])
    states |= set(detail.get("by_domain") or {})
    for rows in (detail.get("trajectory") or {}).values():
        states |= {r.get("state") for r in rows if isinstance(r, dict) and r.get("state")}
    states |= {s.get("state") for s in detail.get("steering") or [] if s.get("state")}
    return sorted(states, key=lambda s: (s != "base", s))


def _finish_packet(mode, neurons, seed, text, main_ws, diff_ws, smap, amap, scale, extra_meta=None):
    allw = main_ws + diff_ws
    pk = {"packet_version": PROMPT_VERSION, "mode": mode, "neurons": neurons, "seed": seed, "created": now(),
          "scale": scale, "state_map": smap, "adapter_map": amap,
          "windows": [{"wid": w["wid"], "ex_id": w["ex_id"], "neuron": w["neuron"], "category": w["category"],
                       "kind": w["kind"], "domain": w["domain"], "focus_token": w["tokens"][w["focus"]],
                       "focus_value": round(_focus_val(w), 4), "true_focus_value": w.get("true_focus_value")}
                      for w in allw],
          "text": text, "est_tokens": _est_tokens(SYSTEM_PROMPT + text)}
    pk.update(extra_meta or {})
    pk["hash"] = _sha({"v": PROMPT_VERSION, "system": SYSTEM_PROMPT, "text": text})
    return pk


def build_packet(detail, mode="blind", n_windows=30, seed=0, index_row=None, include_steering=True,
                 include_lens=True, max_diff_windows=5):
    """Evidence packet for one neuron.

    mode: "blind" (default; hides traits, adapter names, selection reasons, example kinds, prompt ids),
          "unblinded" (shows them, plus training windows), or "shuffled" (control: blind packet with activation
          values permuted across windows and tokens; text kept; aggregate, steering and lens sections dropped).
    """
    assert mode in ("blind", "unblinded", "shuffled")
    blind = mode != "unblinded"
    rng = random.Random(f"{detail['id']}|{seed}|{mode}")
    scale = neuron_scale(detail)
    wins = _discovery_windows(detail, include_training=not blind)
    for w in wins:
        w["category"] = _categorize(w, scale)
    states = _state_list(wins, detail)
    if blind:
        smap, amap = make_state_map(states, f"{detail['id']}|{seed}")
    else:
        smap, amap = {s: s for s in states}, {}
    has_states = any(len(w["acts"]) > 1 for w in wins)
    diff_pool = [w for w in wins if w["kind"] == "top_diff"] if has_states else []
    diff_ws = []
    if diff_pool and max_diff_windows:
        def maxdiff(w):
            b = w["acts"]["base"][w["focus"]]
            return max(abs(v[w["focus"]] - b) for v in w["acts"].values())
        diff_ws = sorted(diff_pool, key=lambda w: -maxdiff(w))[:max_diff_windows]
        for w in diff_ws:
            w["category"] = "state_diff"
    main_pool = [w for w in wins if w not in diff_ws and (w["kind"] != "top_diff" or not has_states)]
    if mode == "unblinded":
        train = [w for w in main_pool if w["kind"] == "training"][:4]
        main_pool = [w for w in main_pool if w["kind"] != "training"]
    main_ws = _select(main_pool, n_windows - len(diff_ws), rng)
    if mode == "unblinded":
        main_ws += train
    if mode == "shuffled":
        _shuffle_acts(main_ws + diff_ws, rng)
        for w in main_ws:
            w["category"] = "random" if w["kind"] == "random" else _categorize(w, scale)

    state_names = sorted([(s, "base" if s == "base" else f"state {smap[s]}") for s in states
                          if any(s in w["acts"] for w in main_ws + diff_ws)], key=lambda x: (x[0] != "base", x[1]))
    header, extra = None, []
    # loop to trim if over the token target
    while True:
        _sort_and_number(main_ws, diff_ws, rng)
        header = [f"# Evidence packet ({len(main_ws) + len(diff_ws)} windows)",
                  f"Neuron activation scale: the strongest discovery activation magnitude is {scale:.2f}; "
                  f"'strong' below means |act| >= {STRONG_FRAC * scale:.2f}."]
        if len(state_names) > 1:
            header.append("Measured states: " + ", ".join(n for _, n in state_names)
                          + ". 'base' is the unmodified model; lettered states are modified versions of it.")
        if mode == "unblinded" and index_row:
            header.append(f"Selected for: {index_row.get('selected_for')} group={index_row.get('group')} "
                          f"reason={index_row.get('selection_reason')}")
        extra = []
        if mode != "shuffled":
            extra.append(("Separate evidence: aggregate differences between model states (identical tokens)",
                          _adapter_section(detail, index_row, smap, blind)))
            if include_steering:
                extra.append(("Separate evidence: steering excerpts (the neuron's activation was modified while "
                              "generating; this is output-side evidence)", _steering_section(detail, smap, blind)))
            lens = (detail.get("lens") or {}).get("base") or {}
            if include_lens and (lens.get("up") or lens.get("down")):
                extra.append(("Separate evidence: weight-lens tokens (approximate, from weights only)", [
                    "- input side (tokens whose embeddings align with the neuron's input weights): "
                    + ", ".join(json.dumps(t) for t in (lens.get("up") or [])[:15]),
                    "- output side (tokens the neuron's output pushes toward): "
                    + ", ".join(json.dumps(t) for t in (lens.get("down") or [])[:15])]))
        text = _assemble(header, main_ws, diff_ws, state_names, scale, mode == "unblinded", extra)
        if _est_tokens(SYSTEM_PROMPT + text) <= TARGET_TOKENS[1] or len(main_ws) + len(diff_ws) <= 20:
            break
        # drop the weakest moderate/random window, else the last high window
        for cat in ("moderate", "random", "high_neg", "high_pos"):
            cands = [w for w in main_ws if w["category"] == cat]
            if len(cands) > 2 or cat == "high_pos":
                main_ws.remove(cands[-1])
                break
    return _finish_packet(mode, [detail["id"]], seed, text, main_ws, diff_ws, smap, amap, scale)


def _sort_and_number(main_ws, diff_ws, rng):
    order = {"high_pos": 0, "high_neg": 1, "moderate": 2, "random": 3, "training": 4}
    main_ws.sort(key=lambda w: (order.get(w["category"], 5),
                                -_focus_val(w) if w["category"] != "high_neg" else _focus_val(w)))
    for i, w in enumerate(main_ws + diff_ws):
        w["wid"] = f"W{i + 1:02d}"


def _shuffle_acts(ws, rng):
    """Permute activations while keeping text: focus values (all states jointly) are permuted among windows, and all
    non-focus token values are permuted among non-focus positions. Summary statistics are unchanged."""
    for w in ws:
        w["true_focus_value"] = round(_focus_val(w), 4)
    states = sorted({s for w in ws for s in w["acts"]})
    def tup(w, i):
        return tuple(w["acts"][s][i] if s in w["acts"] else None for s in states)
    focus_vals = [tup(w, w["focus"]) for w in ws]
    other = [tup(w, i) for w in ws for i in range(len(w["tokens"])) if i != w["focus"]]
    rng.shuffle(focus_vals)
    rng.shuffle(other)
    k = 0
    for w, fv in zip(ws, focus_vals):
        new = {s: [0.0] * len(w["tokens"]) for s in w["acts"]}
        for i in range(len(w["tokens"])):
            t = fv if i == w["focus"] else other[k]
            if i != w["focus"]:
                k += 1
            for j, s in enumerate(states):
                if s in new:
                    new[s][i] = t[j] if t[j] is not None else 0.0
        w["acts"] = new


def build_ambiguous_packet(detail_a, detail_b, n_windows=30, seed=0):
    """Control: windows from two different neurons presented as one neuron (aggregates/steering/lens dropped)."""
    rng = random.Random(f"{detail_a['id']}|{detail_b['id']}|{seed}|ambiguous")
    halves, scales = [], []
    for d in (detail_a, detail_b):
        sc = neuron_scale(d)
        scales.append(sc)
        ws = [w for w in _discovery_windows(d, False) if w["kind"] != "top_diff"]
        for w in ws:
            w["category"] = _categorize(w, sc)
        halves.append(_select(ws, n_windows // 2, rng))
    # rescale B's activations to A's scale so the magnitudes do not reveal the mix
    r = scales[0] / scales[1]
    for w in halves[1]:
        w["acts"] = {s: [round(v * r, 4) for v in a] for s, a in w["acts"].items()}
    main_ws = halves[0] + halves[1]
    scale = scales[0]
    states = sorted({s for w in main_ws for s in w["acts"]}, key=lambda s: (s != "base", s))
    smap, amap = make_state_map(states, f"{detail_a['id']}|{seed}")
    state_names = sorted([(s, "base" if s == "base" else f"state {smap[s]}") for s in states],
                         key=lambda x: (x[0] != "base", x[1]))
    _sort_and_number(main_ws, [], rng)
    header = [f"# Evidence packet ({len(main_ws)} windows)",
              f"Neuron activation scale: the strongest discovery activation magnitude is {scale:.2f}; "
              f"'strong' below means |act| >= {STRONG_FRAC * scale:.2f}."]
    if len(state_names) > 1:
        header.append("Measured states: " + ", ".join(n for _, n in state_names)
                      + ". 'base' is the unmodified model; lettered states are modified versions of it.")
    text = _assemble(header, main_ws, [], state_names, scale, False, [])
    return _finish_packet("ambiguous", [detail_a["id"], detail_b["id"]], seed, text, main_ws, [], smap, amap, scale,
                          {"rescale_b": r})


# ----------------------------------------------------------------------------------------------------------------
# 2. Labeler call over OpenRouter
# ----------------------------------------------------------------------------------------------------------------

def _api_key():
    """keys.env first (the shell may carry a stale OPENROUTER_API_KEY), then the environment."""
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"].strip()
    raise RuntimeError(f"OPENROUTER_API_KEY not set and not found in {KEYS_FILE}")


def fetch_prices(models=MODELS, save=True):
    """Current OpenRouter prices (USD per token) for `models`; saved with a timestamp to logs/openrouter_prices.json."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "persona-neurons"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)["data"]
    out = {m["id"]: {k: float(v) for k, v in m["pricing"].items() if _isnum(v)} for m in data if m["id"] in models}
    if save:
        _write_json(PRICES_FILE, {"checked": now(), "unit": "USD per token", "prices": out})
    return out


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _prices():
    if PRICES_FILE.exists():
        try:
            return {**FALLBACK_PRICES, **json.loads(PRICES_FILE.read_text())["prices"]}
        except Exception:
            pass
    return FALLBACK_PRICES


def estimate_cost(model, in_tokens, out_tokens):
    p = _prices().get(model, {"prompt": 1e-6, "completion": 2e-6})
    return in_tokens * p["prompt"] + out_tokens * p["completion"]


def validate(obj, schema, path="$"):
    """Minimal JSON-schema check (type/enum/required/properties/items/additionalProperties). Returns error list."""
    errs = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "number": (int, float), "integer": int}
    if t and not isinstance(obj, types[t]) or (t in ("number", "integer") and isinstance(obj, bool)):
        return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing {k}")
        props = schema.get("properties", {})
        for k, v in obj.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k}")
    if t == "array" and "items" in schema:
        for i, v in enumerate(obj):
            errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


def _test_texts(t):
    """Flat labeler test (a_/b_/c_ fields) -> list of {"text", "expected", "target"}; passes through "texts"."""
    if "texts" in t:
        return t["texts"]
    return [{"text": t[f"{x}_text"], "expected": t[f"{x}_expected"], "target": t.get(f"{x}_target", "")}
            for x in "abc" if t.get(f"{x}_text") and t.get(f"{x}_expected")]


def validate_labeler_output(obj):
    if isinstance(obj, dict):  # the optional third text may be omitted entirely
        for t in obj.get("proposed_tests") or []:
            if isinstance(t, dict):
                for f in ("c_text", "c_expected", "c_target"):
                    t.setdefault(f, "")
    errs = validate(obj, LABELER_SCHEMA)
    if errs:
        return errs
    if not obj["alternatives"]:
        errs.append("alternatives: need two")
    for i, t in enumerate(obj["proposed_tests"]):
        if len(_test_texts(t)) < 2:
            errs.append(f"proposed_tests[{i}]: need at least texts a and b")
    if not obj["proposed_tests"]:
        errs.append("proposed_tests: empty")
    return errs


def _parse_json(content):
    if content is None:
        raise ValueError("empty content")
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    i = s.find("{")
    if i < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(s[i:])  # ignores trailing text after the first complete object
    return obj


def _urlopen_deadline(req, timeout):
    """urlopen with a wall-clock deadline (OpenRouter keep-alive bytes defeat the socket timeout). A timed-out
    request may still complete and be billed upstream; compare logs with the key's usage if that matters."""
    import concurrent.futures as cf

    def go():
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(go)
    try:
        return fut.result(timeout=timeout)
    except cf.TimeoutError:
        raise TimeoutError from None
    finally:
        ex.shutdown(wait=False)


def _post(body, timeout=120, tries=3):
    """POST with backoff on 429/5xx (those responses are not billed)."""
    for k in range(tries):
        req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/persona-neurons", "X-Title": "persona-neurons explain"})
        try:
            return _urlopen_deadline(req, timeout)
        except TimeoutError:
            if k < tries - 1:
                print(f"request exceeded {timeout}s wall clock; retrying", file=sys.stderr)
                continue
            raise RuntimeError(f"OpenRouter request exceeded {timeout}s wall clock {tries} times") from None
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if (e.code == 429 or e.code >= 500) and k < tries - 1:
                time.sleep(5 * 2 ** k)
                continue
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {msg}") from None


def _log_usage(rec):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _usage_of(resp, model):
    u = resp.get("usage") or {}
    inp, out = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
    reas = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
    cost = u.get("cost")
    return {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": reas,
            "cached_tokens": ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0,
            "cost_usd": float(cost) if cost is not None else estimate_cost(model, inp, out),
            "cost_source": "openrouter" if cost is not None else "estimated"}


def call_json(model, system, user, schema, validator, purpose, cache_key, max_tokens=MAX_TOKENS, temperature=0.0,
              schema_name="neuron_card", use_cache=True, extra_log=None, refresh_invalid=False):
    """One structured-JSON call with one retry on invalid output. Cached under logs/explain_cache by cache_key+model.

    Returns {"parsed", "content", "valid", "errors", "usage", "attempts", "cached", "model"}.
    """
    cpath = CACHE_DIR / f"{cache_key[:32]}__{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.json"
    if use_cache and cpath.exists():
        res = json.loads(cpath.read_text())
        if not res.get("valid"):  # re-check cached raw outputs with the current parser/validator (free)
            for a in res.get("attempts", []):
                try:
                    obj = _parse_json(a["content"])
                    if not validator(obj):
                        res.update(parsed=obj, content=a["content"], valid=True, errors=[], revalidated=now())
                        _write_json(cpath, res)
                        break
                except Exception:
                    pass
        if res.get("valid") or not refresh_invalid:
            res["cached"] = True
            return res
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    total = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0}
    attempts, fmt = [], RESPONSE_FORMAT.get(model, "json_schema")
    parsed, errors, content = None, ["not called"], None
    for attempt in range(2):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "reasoning": {"enabled": False}, "usage": {"include": True}}
        if fmt == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        else:
            body["response_format"] = {"type": "json_object"}
        try:
            resp = _post(body)
        except RuntimeError as e:
            if fmt == "json_schema" and ("response_format" in str(e) or "No endpoints" in str(e) or "400" in str(e)):
                fmt = "json_object"  # fall back once; not counted as an attempt
                body["response_format"] = {"type": "json_object"}
                resp = _post(body)
            else:
                raise
        ch = (resp.get("choices") or [{}])[0]
        content = (ch.get("message") or {}).get("content")
        u = _usage_of(resp, model)
        for k in total:
            total[k] += u[k]
        try:
            parsed = _parse_json(content)
            errors = validator(parsed)
        except Exception as e:
            parsed, errors = None, [f"parse error: {e}"]
        rec = {"ts": now(), "purpose": purpose, "model": model, "provider": resp.get("provider"),
               "generation_id": resp.get("id"), "attempt": attempt + 1, "finish_reason": ch.get("finish_reason"),
               "valid": not errors, "response_format": fmt, "cache_key": cache_key[:32], **u, **(extra_log or {})}
        _log_usage(rec)
        attempts.append({"content": content, "errors": errors, "finish_reason": ch.get("finish_reason"),
                         "usage": u, "provider": resp.get("provider"), "generation_id": resp.get("id")})
        if not errors:
            break
        if ch.get("finish_reason") == "length" or not content:
            # truncated: resend the original request with a brevity instruction (not the cut-off text)
            messages = messages[:2] + [{"role": "user", "content": "Your previous answer was cut off by the length "
                                        "limit. Answer again as minified JSON, much more briefly: short hypotheses, "
                                        "2 texts per test, at most 10 words per text."}]
        else:
            messages = messages[:2] + [{"role": "assistant", "content": content},
                                       {"role": "user", "content": "That output was invalid: " + "; ".join(errors[:5])
                                        + ". Return only one complete minified JSON object matching the schema."}]
    res = {"model": model, "parsed": parsed, "content": content, "valid": not errors, "errors": errors,
           "usage": total, "attempts": attempts, "cached": False, "created": now(), "purpose": purpose}
    _write_json(cpath, res)
    return res


def label_packet(packet, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS, use_cache=True, refresh_invalid=False):
    """Run the labeler on a packet. Returns the call result plus a decoded card version (ex_ids, checks)."""
    res = call_json(model, SYSTEM_PROMPT, packet["text"], LABELER_SCHEMA, validate_labeler_output,
                    purpose=f"explain:{packet['mode']}", cache_key=packet["hash"], max_tokens=max_tokens,
                    use_cache=use_cache, refresh_invalid=refresh_invalid, extra_log={"packet_hash": packet["hash"][:16], "neurons": packet["neurons"]})
    res["version"] = to_card_version(res, packet) if res["valid"] else None
    res["checks"] = evidence_checks(res["parsed"], packet) if res["valid"] else None
    return res


def to_card_version(res, packet):
    p = res["parsed"]
    wmap = {w["wid"]: w["ex_id"] for w in packet["windows"]}

    def decode(ids):
        good, bad = [], []
        for i in ids:
            i = i.strip()
            (good if i in wmap else bad).append(wmap.get(i, i))
        return good, bad
    sup, bad1 = decode(p["supporting"])
    con, bad2 = decode(p["contradicting"])
    tests = []
    for k, t in enumerate(p["proposed_tests"]):
        tests.append({"test_id": f"t{k + 1}", "discriminates": t["discriminates"], "texts": _test_texts(t)})
    return {
        "author": res["model"], "created": res.get("created", now()), "kind": "automated",
        "blind": packet["mode"] != "unblinded", "label": p["label"], "trigger_hypothesis": p["trigger_hypothesis"],
        "output_hypothesis": p["output_hypothesis"], "alternatives": p["alternatives"][:2],
        "adapter_changes": p["adapter_changes"], "supporting": sup, "contradicting": con,
        "proposed_tests": [f"{t['discriminates']} :: " + " | ".join(f"[{x['expected']}] {x['text']}" for x in t["texts"])
                           for t in tests],
        "proposed_tests_detail": tests, "confidence": p["confidence"],
        "no_coherent_hypothesis": p["no_coherent_hypothesis"],
        "usage": {k: res["usage"][k] for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cost_usd")},
        "invalid_citations": bad1 + bad2, "packet_hash": packet["hash"], "packet_mode": packet["mode"],
        "state_map": packet["state_map"], "prompt_version": PROMPT_VERSION,
    }


def evidence_checks(p, packet):
    """Automatic evidence-use checks for manual review (not a quality score)."""
    cat = {w["wid"]: w for w in packet["windows"]}
    sup = [i for i in p["supporting"] if i in cat]
    con = [i for i in p["contradicting"] if i in cat]
    bad = [i for i in p["supporting"] + p["contradicting"] if i not in cat]
    strong = [i for i in sup if cat[i]["category"] in ("high_pos", "high_neg", "state_diff")]
    sup_cats = Counter(cat[i]["category"] for i in sup)
    tts = [_test_texts(t) for t in p["proposed_tests"]]
    n_contrast = sum(1 for tx in tts if len({x["expected"] for x in tx}) >= 2)
    n_target_ok = sum(1 for tx in tts for x in tx
                      if x["expected"] == "low" or (x["target"] and x["target"].lower() in x["text"].lower()))
    n_texts = sum(len(tx) for tx in tts)
    by_neuron = Counter(cat[i]["neuron"] for i in sup)
    return {"n_supporting": len(sup), "n_contradicting": len(con), "invalid_ids": bad,
            "supporting_strong_frac": round(len(strong) / len(sup), 2) if sup else None,
            "supporting_categories": dict(sup_cats), "supporting_by_neuron": dict(by_neuron),
            "n_tests": len(p["proposed_tests"]), "n_contrast_tests": n_contrast,
            "targets_found_frac": round(n_target_ok / n_texts, 2) if n_texts else None,
            "mentions_negative": bool(re.search(r"negativ", p["trigger_hypothesis"] + p["label"], re.I)),
            "no_coherent": p["no_coherent_hypothesis"], "confidence": p["confidence"]}


def load_card(run_dir, neuron):
    p = Path(run_dir) / "neurons" / "cards" / (nid_file(neuron) + ".json")
    if p.exists():
        return json.loads(p.read_text())
    return {"id": neuron, "status": "untested", "versions": [], "tests": [], "disagreements": []}


def save_card(run_dir, card):
    _write_json(Path(run_dir) / "neurons" / "cards" / (nid_file(card["id"]) + ".json"), card)


def append_card_version(run_dir, neuron, version):
    """Append a version (never overwrites existing versions). Returns the version number."""
    card = load_card(run_dir, neuron)
    v = dict(version)
    v["v"] = max([x.get("v", 0) for x in card["versions"]] + [0]) + 1
    card["versions"].append(v)
    save_card(run_dir, card)
    return v["v"]


def note_disagreement(run_dir, neuron, va, vb):
    """Record a disagreement between two automated versions if their evidence use diverges."""
    card = load_card(run_dir, neuron)
    vs = {x["v"]: x for x in card["versions"]}
    a, b = vs[va], vs[vb]
    sa, sb = set(a["supporting"]), set(b["supporting"])
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 1.0
    if a["no_coherent_hypothesis"] != b["no_coherent_hypothesis"] or jac < 0.3:
        card["disagreements"].append(
            f"{now()} v{va} ({a['author']}) '{a['label']}' vs v{vb} ({b['author']}) '{b['label']}': "
            f"supporting overlap {jac:.2f}, no_coherent {a['no_coherent_hypothesis']}/{b['no_coherent_hypothesis']}")
        save_card(run_dir, card)


# ----------------------------------------------------------------------------------------------------------------
# 4. Prediction checker
# ----------------------------------------------------------------------------------------------------------------
# measure_fn(texts: list[str], neuron: str, state: str) -> list[{"tokens": list[str], "acts": list[float]}]
# simulate_fn(version: dict, windows: list[{"id", "available", "focus_token"}]) -> list[float]   (-10..10)

def _get_version(card, v=None):
    autos = [x for x in card["versions"] if x.get("proposed_tests_detail")]
    if not autos:
        raise ValueError("card has no version with structured proposed tests")
    return next(x for x in autos if x["v"] == v) if v else autos[-1]


def validation_windows(detail):
    out = []
    for ex in detail.get("examples", []):
        if ex.get("split") == "validation" or ex.get("kind") == "validation":
            w = _trim(ex)
            out.append({"ex_id": w["ex_id"], "text": "".join(w["tokens"]), "tokens": w["tokens"], "focus": w["focus"],
                        "available": "".join(_tok(t) for t in w["tokens"][:w["focus"]]) + "⟦" + _tok(w["tokens"][w["focus"]]) + "⟧",
                        "focus_token": w["tokens"][w["focus"]],
                        "stored_act": w["acts"]["base"][w["focus"]] if w["acts"].get("base") else None})
    return out


def measurement_plan(card, detail, version=None):
    """Texts to measure for a card version: every proposed-test text plus held-out validation windows (those with
    stored base activations are marked so they need not be remeasured)."""
    ver = _get_version(card, version)
    items = []
    for t in ver["proposed_tests_detail"]:
        for k, x in enumerate(t["texts"]):
            items.append({"item_id": f"v{ver['v']}-{t['test_id']}-{k}", "kind": "test", "test_id": t["test_id"],
                          "text": x["text"], "expected": x["expected"], "target": x.get("target", "")})
    for w in validation_windows(detail):
        items.append({"item_id": w["ex_id"], "kind": "validation", "text": w["text"], "focus_token": w["focus_token"],
                      "stored": w["stored_act"] is not None})
    return items


def _target_tokens(tokens, text, target):
    """Indices of tokens overlapping the first case-insensitive occurrence of target in the joined tokens."""
    joined = "".join(tokens)
    if not target:
        return None
    i = joined.lower().find(target.lower())
    if i < 0:
        i = joined.lower().find(target.strip().lower())
        if i < 0:
            return None
        target = target.strip()
    j, pos, idx = i + len(target), 0, []
    for k, t in enumerate(tokens):
        if pos < j and pos + len(t) > i:
            idx.append(k)
        pos += len(t)
    return idx or None


def _spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def reference_levels(detail):
    """Positive and negative reference magnitudes: median of the five strongest discovery focus values per sign,
    floored at 10% of the overall scale."""
    scale = neuron_scale(detail)
    vals = [ex["acts"]["base"][ex["focus"]] for ex in detail.get("examples", [])
            if ex.get("split", "discovery") == "discovery" and (ex.get("acts") or {}).get("base")]
    pos = sorted([v for v in vals if v > 0], reverse=True)[:5]
    neg = sorted([v for v in vals if v < 0])[:5]
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0
    return {"scale": scale, "pos_ref": max(med(pos), 0.1 * scale), "neg_ref": min(med(neg), -0.1 * scale)}


def score_text(m, x, ref):
    """Score one measured text against its expectation. Returns a card test-text record with numbers."""
    toks, acts = m["tokens"], m["acts"]
    idx = _target_tokens(toks, x["text"], x.get("target", "")) if x["expected"] != "low" else None
    region = idx if idx else list(range(len(toks)))
    vals = [acts[i] for i in region]
    mx, mn = max(acts), min(acts)
    if x["expected"] == "high":
        val, thr = max(vals), 0.5 * ref["pos_ref"]
        ok = val >= thr
    elif x["expected"] == "negative":
        val, thr = min(vals), 0.5 * ref["neg_ref"]
        ok = val <= thr
    else:
        val = max(abs(mx), abs(mn))
        thr = 0.25 * max(ref["pos_ref"], -ref["neg_ref"])
        ok = val <= thr
    return {"text": x["text"], "expected": x["expected"], "target": x.get("target", ""),
            "measured_max": round(mx, 4), "measured_min": round(mn, 4), "value": round(val, 4),
            "threshold": round(thr, 4), "pass": bool(ok),
            "measured_at": "target" if idx else ("token" if x["expected"] == "low" else "whole_text (target not found)")}


def score_test(t, measured, ref):
    recs = [score_text(m, x, ref) for m, x in zip(measured, t["texts"])]
    frac = sum(r["pass"] for r in recs) / len(recs)
    highs = [r["value"] for r in recs if r["expected"] == "high"]
    negs = [r["value"] for r in recs if r["expected"] == "negative"]
    lows_max = [r["measured_max"] for r in recs if r["expected"] == "low"]
    lows_min = [r["measured_min"] for r in recs if r["expected"] == "low"]
    margins = []
    if highs and lows_max:
        margins.append((min(highs) - max(lows_max)) / ref["scale"])
    if negs and lows_min:
        margins.append((min(lows_min) - max(negs)) / ref["scale"])
    margin = min(margins) if margins else None
    if frac == 1.0 and (margin is None or margin > 0):
        result = "pass"
    elif frac < 0.5 or (margin is not None and margin <= 0):
        result = "fail"
    else:
        result = "mixed"
    return recs, result, margin


def check_card(run_dir, neuron, measure_fn, state="base", version=None, simulate_fn=None, write=True,
               source="unspecified"):
    """Score a card version's proposed tests (and optionally held-out validation windows) numerically, append the
    results to card["tests"], and update card["status"]. Returns the new test entries."""
    run_dir = Path(run_dir)
    card, detail = load_card(run_dir, neuron), load_detail(run_dir, neuron)
    ver = _get_version(card, version)
    ref = reference_levels(detail)
    tests = ver["proposed_tests_detail"]
    texts = [x["text"] for t in tests for x in t["texts"]]
    measured = measure_fn(texts, neuron, state) if texts else []
    entries, k = [], 0
    for t in tests:
        n = len(t["texts"])
        recs, result, margin = score_test(t, measured[k:k + n], ref)
        k += n
        entries.append({"test_id": f"v{ver['v']}-{t['test_id']}", "card_version": ver["v"],
                        "description": t["discriminates"], "prediction": "; ".join(
                            f"{x['expected']} on {x.get('target') or 'whole text'!r}" for x in t["texts"]),
                        "texts": recs, "result": result, "state": state,
                        "metrics": {"frac_texts_pass": round(sum(r["pass"] for r in recs) / n, 3),
                                    "contrast_margin_in_scale": None if margin is None else round(margin, 3),
                                    **{k2: round(v2, 4) for k2, v2 in ref.items()}},
                        "notes": f"measure={source}", "created": now()})
    if simulate_fn is not None:
        vw = validation_windows(detail)
        need = [w for w in vw if w["stored_act"] is None]
        if need:
            ms = measure_fn([w["text"] for w in need], neuron, state)
            for w, m in zip(need, ms):
                # first token of the remeasured text matching the stored focus token text from the left
                w["stored_act"] = m["acts"][min(w["focus"], len(m["acts"]) - 1)]
        if vw:
            preds = simulate_fn(ver, [{"id": f"V{i + 1}", "available": w["available"], "focus_token": w["focus_token"]}
                                      for i, w in enumerate(vw)])
            actual = [w["stored_act"] for w in vw]
            rho = _spearman(preds, actual)
            strong = [(p, a) for p, a in zip(preds, actual) if abs(a) >= STRONG_FRAC * ref["scale"]]
            sign_acc = (sum(1 for p, a in strong if p * a > 0) / len(strong)) if strong else None
            result = "pass" if rho is not None and rho >= 0.5 else ("fail" if rho is None or rho < 0.2 else "mixed")
            entries.append({"test_id": f"v{ver['v']}-validation", "card_version": ver["v"],
                            "description": "held-out validation windows: simulated vs measured activation at focus",
                            "prediction": "simulated activations from the card's hypothesis rank-correlate with measured",
                            "texts": [{"text": w["text"], "ex_id": w["ex_id"], "expected": f"sim={p:+.1f}",
                                       "measured_max": round(a, 4), "measured_at": "token"}
                                      for w, p, a in zip(vw, preds, actual)],
                            "result": result, "state": state,
                            "metrics": {"spearman": None if rho is None else round(rho, 3), "n": len(vw),
                                        "sign_acc_on_strong": None if sign_acc is None else round(sign_acc, 3),
                                        "n_strong": len(strong)},
                            "notes": f"simulator={getattr(simulate_fn, '__name__', 'fn')}", "created": now()})
    if write:
        card = load_card(run_dir, neuron)  # reload in case of concurrent edits
        card["tests"].extend(entries)
        card["status"] = card_status([e for e in card["tests"] if e.get("card_version") == ver["v"]])
        save_card(run_dir, card)
    return entries


def card_status(entries):
    if not entries:
        return "untested"
    val = [e for e in entries if e["test_id"].endswith("validation")]
    tests = [e for e in entries if not e["test_id"].endswith("validation")]
    frac = sum(e["result"] == "pass" for e in tests) / len(tests) if tests else None
    vfail = any(e["result"] == "fail" for e in val)
    if (frac is not None and frac <= 0.3) or (vfail and (frac is None or frac < 0.7)):
        return "rejected"
    if (frac is None or frac >= 0.7) and not vfail:
        return "tested"
    return "mixed"


SIM_SYSTEM = """You predict a language-model neuron's activation from a written hypothesis about it.
For each item, the model has read the AVAILABLE text up to and including the marked token ⟦like this⟧. Predict the
neuron's activation at the marked token on a scale from -10 (strongest negative) through 0 (near zero) to 10
(strongest positive), using only the hypothesis. Return JSON: {"predictions": [{"id": "V1", "value": 0}, ...]} with
one entry per item."""

SIM_SCHEMA = {"type": "object", "properties": {"predictions": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "value": {"type": "number"}},
    "required": ["id", "value"], "additionalProperties": False}}}, "required": ["predictions"],
    "additionalProperties": False}


def llm_simulator(model=DEFAULT_MODEL, max_tokens=800):
    """simulate_fn backed by an OpenRouter model (auxiliary to numerical tests; logged and cached like labels)."""
    def simulate(ver, windows):
        hyp = (f"Label: {ver['label']}\nTrigger hypothesis: {ver['trigger_hypothesis']}\n"
               f"No coherent hypothesis: {ver['no_coherent_hypothesis']}")
        items = "\n".join(f"{w['id']}: {w['available']}" for w in windows)
        user = f"{hyp}\n\nItems:\n{items}"
        res = call_json(model, SIM_SYSTEM, user, SIM_SCHEMA, lambda o: validate(o, SIM_SCHEMA), "explain:simulate",
                        _sha({"sim": SIM_SYSTEM, "u": user}), max_tokens=max_tokens, schema_name="simulation")
        got = {p["id"]: float(p["value"]) for p in (res["parsed"] or {}).get("predictions", [])}
        return [got.get(w["id"], 0.0) for w in windows]
    simulate.__name__ = f"llm_simulator:{model}"
    return simulate


# ----------------------------------------------------------------------------------------------------------------
# 5. Pilot runner
# ----------------------------------------------------------------------------------------------------------------

def _one_line(s, n=160):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def run_pilot(run, neuron_ids, models=MODELS, shuffled_for=None, ambiguous_pairs=None, controls=True,
              n_windows=30, budget_usd=2.0, write_cards=True, out_dir=None, max_tokens=MAX_TOKENS):
    """Blind packets for each neuron x each model, plus shuffled controls (default: every neuron) and ambiguous
    controls (default: consecutive pairs). Writes everything to runs/<id>/explain_pilot/ and summary.md.
    Cards are appended only for real (blind) packets; control outputs stay in the pilot directory."""
    run_dir = resolve_run(run)
    out = Path(out_dir) if out_dir else run_dir / "explain_pilot"
    out.mkdir(parents=True, exist_ok=True)
    index = load_index(run_dir)
    details = {n: load_detail(run_dir, n) for n in neuron_ids}
    jobs = [(n, build_packet(details[n], "blind", n_windows, index_row=index.get(n))) for n in neuron_ids]
    if controls:
        for n in (neuron_ids if shuffled_for is None else shuffled_for):
            jobs.append((n, build_packet(details.setdefault(n, load_detail(run_dir, n)), "shuffled", n_windows)))
        pairs = ambiguous_pairs if ambiguous_pairs is not None else \
            [(neuron_ids[i], neuron_ids[i + 1]) for i in range(0, len(neuron_ids) - 1, 2)]
        for a, b in pairs:
            jobs.append((f"{a}+{b}", build_ambiguous_packet(details.setdefault(a, load_detail(run_dir, a)),
                                                            details.setdefault(b, load_detail(run_dir, b)), n_windows)))
    spent, rows = 0.0, []
    for name, pk in jobs:
        stem = f"{nid_file(name).replace('+', '__')}__{pk['mode']}"
        _write_json(out / f"{stem}.packet.json", pk)
        (out / f"{stem}.packet.txt").write_text(SYSTEM_PROMPT + "\n\n=====\n\n" + pk["text"])
        versions = []
        for model in models:
            worst = 2 * estimate_cost(model, pk["est_tokens"], max_tokens)
            if spent + worst > budget_usd:
                print(f"budget stop before {name} {pk['mode']} {model}: spent {spent:.4f}", file=sys.stderr)
                rows.append((name, pk, model, None))
                continue
            try:
                res = label_packet(pk, model, max_tokens, refresh_invalid=True)
            except Exception as e:  # record and continue
                res = {"model": model, "error": str(e), "parsed": None, "usage": {"cost_usd": 0.0}, "cached": False}
            if not res.get("cached"):
                spent += res["usage"]["cost_usd"]
            _write_json(out / f"{stem}__{re.sub(r'[^A-Za-z0-9.-]', '_', model)}.json", res)
            if write_cards and pk["mode"] == "blind" and res.get("version"):
                card = load_card(run_dir, name)
                dup = any(v.get("packet_hash") == pk["hash"] and v.get("author") == model for v in card["versions"])
                if not dup:
                    versions.append(append_card_version(run_dir, name, res["version"]))
            rows.append((name, pk, model, res))
            print(f"{name:24s} {pk['mode']:9s} {model:30s} valid={res.get('valid')} cached={res.get('cached')} "
                  f"cost={res['usage']['cost_usd']:.5f} label={_one_line((res.get('parsed') or {}).get('label', res.get('error')), 60)}")
        if len(versions) == 2:
            note_disagreement(run_dir, name, *versions)
    (out / "summary.md").write_text(pilot_summary(rows, spent))
    print(f"spent this run (uncached calls): USD {spent:.5f}; summary: {out / 'summary.md'}")
    return rows, spent


def pilot_summary(rows, spent):
    L = ["# Labeler pilot: side-by-side", "",
         f"Generated {now()}. Uncached spend in this invocation: USD {spent:.5f}. Prompt version {PROMPT_VERSION}.", "",
         "Automatic checks are aids for the manual review, not scores. 'strong frac' = share of cited supporting "
         "windows that are strong positive/negative in the packet the labeler saw. For shuffled controls the "
         "honest answer is low confidence or no coherent hypothesis; for ambiguous controls the labeler should "
         "notice two unrelated clusters (see 'cites by neuron').", ""]
    by_packet = defaultdict(list)
    for name, pk, model, res in rows:
        by_packet[(name, pk["mode"], pk["hash"])].append((model, res, pk))
    for (name, mode, _), items in by_packet.items():
        pk = items[0][2]
        L += [f"## {name} [{mode}]", "",
              f"{len(pk['windows'])} windows, ~{pk['est_tokens']} input tokens, categories "
              f"{dict(Counter(w['category'] for w in pk['windows']))}", ""]
        L.append("| field | " + " | ".join(m for m, _, _ in items) + " |")
        L.append("|---|" + "---|" * len(items))

        def row(label, f):
            vals = []
            for _, res, _ in items:
                try:
                    vals.append(_one_line(f(res), 220).replace("|", "/"))
                except Exception:
                    vals.append("(none)")
            L.append(f"| {label} | " + " | ".join(vals) + " |")
        row("valid / cached", lambda r: f"{r.get('valid')} / {r.get('cached')}" + (f" ERR {r['error']}" if r.get("error") else ""))
        row("label", lambda r: r["parsed"]["label"])
        row("trigger", lambda r: r["parsed"]["trigger_hypothesis"])
        row("output effect", lambda r: r["parsed"]["output_hypothesis"])
        row("alternatives", lambda r: " // ".join(r["parsed"]["alternatives"]))
        row("adapter changes", lambda r: r["parsed"]["adapter_changes"])
        row("confidence / no_coherent", lambda r: f"{r['parsed']['confidence']} / {r['parsed']['no_coherent_hypothesis']}")
        row("supporting (n, strong frac)", lambda r: f"{r['checks']['n_supporting']}, {r['checks']['supporting_strong_frac']} {r['checks']['supporting_categories']}")
        row("contradicting (n)", lambda r: f"{r['checks']['n_contradicting']}: {', '.join(r['parsed']['contradicting'][:8])}")
        row("invalid ids", lambda r: ", ".join(r["checks"]["invalid_ids"]) or "none")
        if mode == "ambiguous":
            row("cites by neuron", lambda r: r["checks"]["supporting_by_neuron"])
        row("tests (n, contrast, targets found)", lambda r: f"{r['checks']['n_tests']}, {r['checks']['n_contrast_tests']}, {r['checks']['targets_found_frac']}")
        row("first two tests", lambda r: " // ".join(
            f"{t['discriminates']}: " + "; ".join(f"[{x['expected']}] {x['text']}" for x in _test_texts(t))
            for t in r["parsed"]["proposed_tests"][:2]))
        row("tokens in/out/reasoning, cost", lambda r: f"{r['usage']['input_tokens']}/{r['usage']['output_tokens']}/"
            f"{r['usage']['reasoning_tokens']}, ${r['usage']['cost_usd']:.5f}")
        L += ["", "Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  "
              "[ ] tests discriminate  Notes:", ""]
    return "\n".join(L)


def usage_total(purpose_prefix="explain"):
    tot, n = 0.0, 0
    if USAGE_LOG.exists():
        for line in USAGE_LOG.read_text().splitlines():
            r = json.loads(line)
            if str(r.get("purpose", "")).startswith(purpose_prefix):
                tot += r.get("cost_usd") or 0.0
                n += 1
    return tot, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prices")
    p = sub.add_parser("packet")
    p.add_argument("run"); p.add_argument("neuron")
    p.add_argument("--mode", default="blind", choices=["blind", "unblinded", "shuffled"])
    p.add_argument("--other", help="second neuron: build an ambiguous control packet")
    p.add_argument("--n", type=int, default=30)
    p = sub.add_parser("label")
    p.add_argument("run"); p.add_argument("neuron")
    p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--no-card", action="store_true")
    p.add_argument("--n", type=int, default=30)
    p = sub.add_parser("pilot")
    p.add_argument("run"); p.add_argument("neurons", nargs="+")
    p.add_argument("--models", default=",".join(MODELS)); p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--no-controls", action="store_true"); p.add_argument("--no-cards", action="store_true")
    p.add_argument("--shuffled-for", default=None, help="comma list; default all neurons")
    p.add_argument("--n", type=int, default=30)
    sub.add_parser("usage")
    a = ap.parse_args(argv)
    if a.cmd == "prices":
        print(json.dumps(fetch_prices(), indent=1))
    elif a.cmd == "usage":
        tot, n = usage_total()
        print(f"explain calls logged: {n}, total USD {tot:.5f}")
    elif a.cmd == "packet":
        rd = resolve_run(a.run)
        d = load_detail(rd, a.neuron)
        pk = build_ambiguous_packet(d, load_detail(rd, a.other), a.n) if a.other else \
            build_packet(d, a.mode, a.n, index_row=load_index(rd).get(a.neuron))
        print(pk["text"])
        print(f"\n[{len(pk['windows'])} windows, ~{pk['est_tokens']} tokens, hash {pk['hash'][:12]}]", file=sys.stderr)
    elif a.cmd == "label":
        rd = resolve_run(a.run)
        pk = build_packet(load_detail(rd, a.neuron), "blind", a.n, index_row=load_index(rd).get(a.neuron))
        res = label_packet(pk, a.model)
        print(json.dumps(res["parsed"], indent=1))
        if res.get("version") and not a.no_card:
            print("appended card version", append_card_version(rd, a.neuron, res["version"]), file=sys.stderr)
    elif a.cmd == "pilot":
        sf = a.shuffled_for.split(",") if a.shuffled_for else None
        run_pilot(a.run, a.neurons, a.models.split(","), shuffled_for=sf, controls=not a.no_controls,
                  n_windows=a.n, budget_usd=a.budget, write_cards=not a.no_cards)


if __name__ == "__main__":
    main()
