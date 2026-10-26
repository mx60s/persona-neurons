"""Behavioral interventions on single neurons, for the detailed profiles.

    python -m pn.steer_eval --run runs/v1 --neurons L33/N7644,... --out runs/v1/behavior/interventions.jsonl

For each neuron, on the first 20 prompts of its trait's saved set (greedy, 160 tokens):
  * additive steering in the base state at doses k * r, k in {-4, -2, +2, +4}, where r is the neuron's natural scale
    (largest of |p99| and |p01| of its base activation on the corpus), applied at every position;
  * zero ablation in the base state and in the trait's amplifier+1 state.
Layer-matched random neurons get the same conditions with their own r. Every row carries the model state and the
intervention separately (docs/run_format.md); score with pn.judge.
"""
import argparse
import json
import os
import random

import numpy as np  # noqa: F401

from pn.behavior import generate
from pn.intervene import describe
from pn.measure import Measurer

AMP = {"openness": "o_amp+1", "neuroticism": "n_amp+1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--prompts", default="data/behavior_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--doses", default="-4,-2,2,4")
    args = ap.parse_args()
    index = {r["id"]: r for r in json.load(open(os.path.join(args.run, "neurons", "index.json")))}
    prompts = json.load(open(args.prompts))
    m = Measurer()
    sm = m.sm
    rng = random.Random(1)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["prompt_id"], r["state"], json.dumps(r["intervention"], sort_keys=True)))
    jobs = []
    for nid in args.neurons.split(","):
        rec = index[nid]
        det = json.load(open(os.path.join(args.run, "neurons", "detail", nid.replace("/", "_") + ".json")))
        trait = next((t for t in ("openness", "neuroticism") if t in rec["selected_for"]), "openness")
        r_ = max(abs(det["quantiles_base"]["p99"]), abs(det["quantiles_base"]["p01"]))
        jobs.append((nid, trait, r_, "candidate"))
        for _ in range(args.n_random):
            n = rng.randrange(sm.n_inter)
            rid = f"L{rec['layer']}/N{n}"
            # natural scale of the random neuron: 99th percentile of |activation| over ten corpus windows
            texts = ["".join(e["tokens"]) for e in det["examples"] if e["kind"] != "training"][:10]
            vals = sorted(abs(x) for a in m.acts(texts, rid, "base") for x in a["acts"][1:])
            jobs.append((rid, trait, vals[int(0.99 * (len(vals) - 1))] or 1e-3, f"random_for:{nid}"))
    with open(args.out, "a") as f:
        for nid, trait, r_, role in jobs:
            layer, n = int(nid.split("/")[0][1:]), int(nid.split("/")[1][1:])
            ps = [p for p in prompts if p["trait"] == trait][:args.n_prompts]
            conds = [("base", [(layer, n, "add", float(k) * r_)], f"add {k} x natural scale") for k in map(float, args.doses.split(","))]
            conds += [("base", [(layer, n, "zero", 0.0)], "zero"), (AMP[trait], [(layer, n, "zero", 0.0)], "zero")]
            for state, edits, label in conds:
                desc = describe(state, edits, unit="act")
                desc["intervention"]["label"] = label
                desc["intervention"]["natural_scale"] = r_
                desc["intervention"]["role"] = role
                todo = [p for p in ps if (p["id"], state, json.dumps(desc["intervention"], sort_keys=True)) not in done]
                if not todo:
                    continue
                res = generate(sm, todo, state, edits=edits, batch=20)
                for p, (text, nt, fin) in zip(todo, res):
                    f.write(json.dumps({"prompt_id": p["id"], **desc, "text": text, "n_tokens": nt, "finish": fin,
                                        "seed": 0, "decoding": {"do_sample": False, "max_new_tokens": 160}}) + "\n")
                f.flush()
                print(nid, role, state, label, len(todo), flush=True)


if __name__ == "__main__":
    main()
