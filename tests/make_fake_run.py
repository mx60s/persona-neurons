"""Writes a synthetic run that follows docs/run_format.md, for exercising the browser without real data.

    python tests/make_fake_run.py --tok ../work/tok --out runs/fake [--layers 4 --inter 192]

Neuron ids fall inside the tiny fixture's range (tests/make_tiny.py: 4 layers x 192 neurons) so live panels work
against it. All numbers are random; nothing here is a measurement.
"""
import argparse
import json
import os
import random
import shutil

STATES = [
    ("base", "base", None),
    ("o_amp-1", "openness amplifier x-1", "openness"), ("o_amp-0.5", "openness amplifier x-0.5", "openness"),
    ("o_amp+0.5", "openness amplifier x0.5", "openness"), ("o_amp+1", "openness amplifier x1", "openness"),
    ("o_sup+1", "openness suppressor x1", "openness"),
    ("n_amp-1", "neuroticism amplifier x-1", "neuroticism"), ("n_amp-0.5", "neuroticism amplifier x-0.5", "neuroticism"),
    ("n_amp+0.5", "neuroticism amplifier x0.5", "neuroticism"), ("n_amp+1", "neuroticism amplifier x1", "neuroticism"),
    ("n_sup+1", "neuroticism suppressor x1", "neuroticism"),
    ("ctrl+1", "neutral control x1", None),
]
DOMAINS = ["chat", "narrative", "factual", "code", "neutral", "emotional"]
TEXTS = [
    ("chat", "User: I keep worrying that the results will come back bad. Assistant: Waiting is hard. It may help to "
             "plan what you would do in each case, so the uncertainty feels smaller."),
    ("narrative", "She opened the strange box slowly, curious about the faint humming inside, and found a map of a "
                  "city that did not exist."),
    ("factual", "The octopus has three hearts and blue blood. Two hearts pump blood through the gills, and the third "
                "pumps it to the rest of the body."),
    ("code", "def mean(xs):\n    if not xs:\n        raise ValueError('empty')\n    return sum(xs) / len(xs)\n"),
    ("emotional", "User: My manager criticized my report in front of everyone. Assistant: That sounds embarrassing. "
                  "Criticism in public can sting even when it is fair."),
    ("neutral", "The train leaves at 9:40 from platform 2. Tickets can be bought at the machine near the entrance."),
]
GROUPS = ["shared", "amp_primary", "sup_primary", "cross_trait", "control_assoc", "surprising"]
POLES = ["o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", "ctrl+1"]


def tokenize(tok, text):
    if tok is None:
        import re
        return re.findall(r"\s*\S+", text)
    ids = tok(text, add_special_tokens=False).input_ids
    return [tok.decode([i]) for i in ids]


def r3(x):
    return round(x, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default=None, help="tokenizer dir for real token strings (optional)")
    ap.add_argument("--out", default="runs/fake")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--inter", type=int, default=192)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tok = None
    if args.tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tok)
    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    for d in ("behavior", "corpus", "neurons/detail", "neurons/cards", "profiles", "edits/cards"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    def dump(rel, obj):
        with open(os.path.join(out, rel), "w") as f:
            json.dump(obj, f, indent=1)

    run_id = os.path.basename(os.path.normpath(out))
    adapters = {k: {"repo": "fake/tiny", "revision": "0000000", "path": f"adapters/{k}", "r": 8, "alpha": 8,
                    "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
                for k in ("o_amp", "o_sup", "n_amp", "n_sup", "ctrl")}
    dump("manifest.json", {
        "run_id": run_id, "created": "2026-10-25T12:00:00Z",
        "model": {"id": "tiny-random-qwen3", "revision": "local", "arch": {"n_layers": args.layers, "hidden": 64,
                                                                           "inter": args.inter, "act": "silu"}},
        "adapters": adapters,
        "states": [{"id": s, "label": lab, "trait": t} for s, lab, t in STATES],
        "traits": {
            "openness": {"amp": "o_amp", "sup": "o_sup", "sweep": ["o_amp-1", "o_amp-0.5", "base", "o_amp+0.5", "o_amp+1"],
                         "poles": ["o_amp+1", "o_sup+1"]},
            "neuroticism": {"amp": "n_amp", "sup": "n_sup", "sweep": ["n_amp-1", "n_amp-0.5", "base", "n_amp+0.5", "n_amp+1"],
                            "poles": ["n_amp+1", "n_sup+1"]},
        },
        "control_state": "ctrl+1",
        "corpus": {"n_docs": len(TEXTS), "n_tokens": 2000, "domains": DOMAINS},
        "notes": "SYNTHETIC run from tests/make_fake_run.py. Random numbers, not measurements.",
    })

    # behavior
    prompts = []
    for i in range(6):
        prompts.append({"id": f"o{i+1:02d}", "trait": "openness", "category": "curiosity", "common": i < 2,
                        "text": ["What would you do with a free afternoon in a city you have never visited?",
                                 "How should I think about a hobby that seems pointless?",
                                 "Describe an unusual idea you find interesting.",
                                 "My friend wants to try a strange new diet. Advice?",
                                 "What do you make of abstract art?", "Plan a weekend for me."][i]})
        prompts.append({"id": f"n{i+1:02d}", "trait": "neuroticism", "category": "waiting", "common": i < 2,
                        "text": ["I'm waiting for exam results. How do I cope?",
                                 "My code review got harsh comments. What now?",
                                 "The flight is delayed by three hours.", "I made a mistake at work today.",
                                 "How do you feel about uncertainty?", "Tell me about a calm morning."][i]})
    dump("behavior/prompts.json", prompts)
    with open(os.path.join(out, "behavior/responses.jsonl"), "w") as f, open(os.path.join(out, "behavior/scores.jsonl"), "w") as g:
        for p in prompts:
            for s, _, t in STATES:
                if t not in (None, p["trait"]):
                    continue
                f.write(json.dumps({"prompt_id": p["id"], "state": s, "intervention": None,
                                    "text": f"[synthetic response to {p['id']} in {s}]", "n_tokens": 12, "finish": "eos",
                                    "seed": 0, "decoding": {"do_sample": False, "max_new_tokens": 160}}) + "\n")
                g.write(json.dumps({"prompt_id": p["id"], "state": s, "judge": "none/synthetic", "trait": p["trait"],
                                    "score": rng.randint(1, 7), "coherence": rng.randint(1, 5), "rationale": "synthetic",
                                    "blind_id": f"b{rng.randint(0, 99999)}"}) + "\n")
    dump("behavior/summary.json", {t: {s: {"trait_score": r3(rng.uniform(2, 6)), "coherence": r3(rng.uniform(3, 5)), "n": 6,
                                           "ci90": [2.0, 5.0]} for s, _, st in STATES if st in (None, t)}
                                   for t in ("openness", "neuroticism")})
    with open(os.path.join(out, "corpus/docs.jsonl"), "w") as f:
        for i, (dom, text) in enumerate(TEXTS):
            f.write(json.dumps({"doc_id": f"{dom}-{i:04d}", "source": "synthetic", "domain": dom, "n_tokens": 40,
                                "boundary": 12 if dom in ("chat", "emotional") else None}) + "\n")

    # neurons
    picked, index = set(), []
    plan = []
    for trait in ("openness", "neuroticism"):
        for g in GROUPS[:4]:
            plan += [(trait, g)] * 2
    plan += [("openness", "control_assoc"), ("neuroticism", "surprising")]
    plan += [(None, "random_control")] * max(0, args.n - len(plan))
    statuses = ["untested", "tested", "mixed", "rejected"]
    for k, (trait, g) in enumerate(plan[: args.n]):
        while True:
            layer, n = rng.randrange(args.layers), rng.randrange(args.inter)
            if (layer, n) not in picked:
                picked.add((layer, n))
                break
        nid, fid = f"L{layer}/N{n}", f"L{layer}_N{n}"
        rms = rng.uniform(0.05, 0.4)
        den = max(rms, 0.02)
        effects = {}
        for s, _, t in STATES:
            if s == "base":
                continue
            md = rng.gauss(0, 0.6) * den
            if trait and t == trait and g in ("shared", "amp_primary") and "amp+" in s:
                md = abs(md) + 0.5 * den
            rd = abs(md) + rng.uniform(0, 0.5) * den
            effects[s] = {"mean_diff": r3(md), "rms_diff": r3(rd), "z_mean": r3(md / den), "z_rms": r3(rd / den),
                          "agree": r3(rng.uniform(0.4, 1.0)), "resid_mean_diff": r3(md * 0.1)}
        kind = "random_control" if g == "random_control" else "candidate"
        group = {"openness": None, "neuroticism": None}
        if kind == "candidate":
            group[trait] = g
            if g == "cross_trait":
                other = "neuroticism" if trait == "openness" else "openness"
                group[other] = "cross_trait"
        has_card = k % 3 != 2
        status = statuses[k % 4] if has_card else "untested"
        index.append({
            "id": nid, "layer": layer, "n": n, "kind": kind,
            "selected_for": [t for t, v in group.items() if v],
            "group": group, "rank": {t: k + 1 for t, v in group.items() if v},
            "selection_reason": "synthetic" if kind == "candidate" else "layer-matched random control",
            "base": {"mean": r3(rng.gauss(0, 0.02)), "rms": r3(rms), "frac_pos": 0.5, "frac_neg": 0.5,
                     "q": {"p01": r3(-2 * rms), "p50": 0.0, "p99": r3(3 * rms), "p999": r3(5 * rms)},
                     "max": r3(6 * rms), "min": r3(-3 * rms)},
            "effects": effects, "denominator": r3(den), "card_status": status,
            "label": f"synthetic label {k}" if has_card else "", "detailed": k in (0, 5),
        })

        # detail
        examples = []
        ex_states = ["base", "o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", "ctrl+1"]
        kinds = ["top_pos", "top_neg", "top_diff", "quantile", "random", "training", "validation"]
        for e in range(7):
            dom, text = TEXTS[e % len(TEXTS)]
            toks = tokenize(tok, text)
            focus = rng.randrange(len(toks))
            base_acts = [r3(rng.gauss(0, rms)) for _ in toks]
            base_acts[focus] = r3((6 if kinds[e] != "top_neg" else -3) * rms)
            acts = {"base": base_acts}
            for s in ex_states[1:]:
                shift = effects[s]["mean_diff"]
                acts[s] = [r3(a * rng.uniform(0.6, 1.4) + shift) for a in base_acts]
            ex = {"ex_id": f"{fid}:e{e:02d}", "doc_id": f"{dom}-{e % len(TEXTS):04d}", "domain": dom, "source": "synthetic",
                  "kind": kinds[e], "split": "validation" if e in (4, 6) else "discovery", "tokens": toks,
                  "start": 0, "focus": focus, "boundary": 12 if dom in ("chat", "emotional") and len(toks) > 12 else None,
                  "acts": acts}
            if e == 0:
                ex["gate"] = {"base": [r3(rng.gauss(0, 1)) for _ in toks]}
                ex["up"] = {"base": [r3(rng.gauss(0, 1)) for _ in toks]}
            if kinds[e] == "training":
                ex["train_meta"] = {"adapter": "o_amp", "role": "chosen", "dataset_path": "synthetic/train.jsonl"}
            examples.append(ex)
        traj = {}
        for t, info in (("openness", ("o_amp", "o_sup")), ("neuroticism", ("n_amp", "n_sup"))):
            pts = []
            for s in [f"{info[0]}-1", f"{info[0]}-0.5", "base", f"{info[0]}+0.5", f"{info[0]}+1", f"{info[1]}+1"]:
                md = 0.0 if s == "base" else effects[s]["mean_diff"]
                pts.append({"state": s, "mean": r3(index[-1]["base"]["mean"] + md), "rms": r3(rms + abs(md)),
                            "mean_diff": r3(md), "z_mean": r3(md / den),
                            "ci90": [r3(md / den - 0.15), r3(md / den + 0.15)]})
            traj[t] = pts
        polar = [s for s in POLES]
        detail = {
            "id": nid, "trajectory": traj,
            "by_domain": {s: {d: {"mean_diff": r3(rng.gauss(0, 0.05)), "n_tok": rng.randint(100, 900)} for d in DOMAINS} for s in polar},
            "by_segment": {s: {"prompt": r3(rng.gauss(0, 0.05)), "continuation": r3(rng.gauss(0, 0.05))} for s in polar},
            "weights": {"down_norm": {"base": 1.0, **{s: r3(1 + rng.gauss(0, 0.01)) for s in polar}},
                        "down_cos_to_base": {s: r3(1 - abs(rng.gauss(0, 0.002))) for s in polar},
                        "gate_cos_to_base": {s: r3(1 - abs(rng.gauss(0, 0.002))) for s in polar},
                        "up_cos_to_base": {s: r3(1 - abs(rng.gauss(0, 0.002))) for s in polar}},
            "lens": {s: {"up": tokenize(tok, " calm curious strange worry plan")[:5],
                         "down": tokenize(tok, " never fear dull fixed panic")[:5]} for s in ["base"] + polar},
            "examples": examples,
            "steering": [{"prompt_id": "n01", "state": "base",
                          "intervention": {"neuron": nid, "mode": "add", "value": 0.5, "unit": "resid_p50"},
                          "text": "[synthetic steered text]", "kl_first_token": 0.1, "top_changes": [[" calm", 0.3]]},
                         {"prompt_id": "n01", "state": "n_amp+1",
                          "intervention": {"neuron": nid, "mode": "zero", "value": 0.0, "unit": "x"},
                          "text": "[synthetic ablated text]", "kl_first_token": 0.02, "top_changes": [[" worry", -0.1]]}],
            "ablation": [{"state": "n_amp+1", "intervention": {"neuron": nid, "mode": "zero", "value": 0.0, "unit": "x"},
                          "metric": "trait_score", "value": 4.1, "baseline": 4.6, "n": 20}],
            "natural_range": {"p01": r3(-2 * rms), "p99": r3(3 * rms), "resid_p50": 0.2, "down_norm": 0.17,
                              "add_unit_in_act": r3(0.2 / 0.17)},
        }
        dump(f"neurons/detail/{fid}.json", detail)

        if has_card:
            versions = [{
                "v": 1, "author": "deepseek/deepseek-v3.2", "created": "2026-10-25T12:00:00Z", "kind": "automated",
                "blind": True, "label": f"synthetic label {k}",
                "trigger_hypothesis": "Fires on words about waiting for an uncertain outcome (synthetic).",
                "output_hypothesis": "Promotes reassuring continuations (synthetic).",
                "alternatives": ["Any future-tense verb", "Second-person advice framing"],
                "adapter_changes": "Amplifier raises activation on the same contexts; no new triggers seen (synthetic).",
                "supporting": [f"{fid}:e00", f"{fid}:e02"], "contradicting": [f"{fid}:e03"],
                "proposed_tests": ["Matched pair: waiting for results vs waiting for a bus", "Same sentence, third person"],
                "confidence": "low", "no_coherent_hypothesis": False,
                "usage": {"input_tokens": 6100, "output_tokens": 800, "reasoning_tokens": 0, "cost_usd": 0.0015}}]
            if k % 2 == 1:
                versions.append({**versions[0], "v": 2, "author": "claude-opus-5-5 (review)", "kind": "review",
                                 "created": "2026-10-25T13:00:00Z", "blind": False,
                                 "label": f"synthetic label {k} (revised)", "confidence": "medium"})
            tests = []
            if status != "untested":
                tests.append({"test_id": "t1", "description": "waiting-for-results vs waiting-for-bus",
                              "prediction": "higher on results",
                              "texts": [{"text": "I am waiting for my biopsy results.", "expected": "high", "measured_max": 0.8, "measured_at": " results"},
                                        {"text": "I am waiting for the bus.", "expected": "low", "measured_max": 0.3, "measured_at": " bus"}],
                              "result": {"tested": "pass", "mixed": "mixed", "rejected": "fail"}[status], "state": "base",
                              "notes": "synthetic"})
            dump(f"neurons/cards/{fid}.json", {"id": nid, "status": status, "versions": versions, "tests": tests,
                                               "disagreements": ["Reviewer thinks the trigger is topic, not waiting (synthetic)."] if k % 2 else []})
        if k in (0, 5):
            with open(os.path.join(out, f"profiles/{fid}.md"), "w") as f:
                f.write(f"# {nid}: synthetic profile\n\n**Selection reason:** synthetic.\n\n## Explanation\n\n"
                        "Fires on *waiting* contexts. This is placeholder text.\n\n## Alternative\n\n- topic words\n- "
                        "sentence position\n\n## Next test\n\n`I am waiting for the bus.` vs `I am waiting for results.`\n")
    dump("neurons/index.json", index)
    print(f"wrote {out}: {len(index)} neurons")


if __name__ == "__main__":
    main()
