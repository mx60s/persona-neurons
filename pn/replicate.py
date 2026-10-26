"""Replication runs for the strongest single-neuron intervention results.

    python -m pn.replicate --out runs/v1/behavior/replication.jsonl

1. Zeroing L30/N606 inside n_amp+1 on all 40 neuroticism prompts and the 104 general prompts, against three
   layer-matched random neurons zeroed the same way, plus the unmodified n_amp+1 rows already in responses.jsonl.
2. Zeroing L22/N3214 and L23/N2218 (separately and together) inside n_sup+1 on the 40 neuroticism prompts, against
   two random layer-matched pairs. Outcome measured offline: softener frequency ("a bit", "slightly", "somewhat",
   "a little", "kind of") and the judge.
Every row records the model state and the intervention separately.
"""
import argparse
import json
import random

from pn.behavior import generate
from pn.intervene import describe
from pn.measure import Measurer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/behavior_prompts.json,data/private/cheery_prompts.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompts = [p for path in args.prompts.split(",") for p in json.load(open(path))]
    neu = [p for p in prompts if p["trait"] == "neuroticism"]
    gen = [p for p in prompts if p["trait"] == "general"]
    m = Measurer()
    sm = m.sm
    rng = random.Random(7)
    conds = []
    sets1 = [("candidate", [(30, 606)])] + [(f"random_{k}", [(30, rng.randrange(sm.n_inter))]) for k in range(3)]
    for role, ns in sets1:
        conds.append(("n_amp+1", ns, role, neu + gen, "replicate L30/N606 zero"))
    sets2 = [("candidate_L22", [(22, 3214)]), ("candidate_L23", [(23, 2218)]), ("candidate_both", [(22, 3214), (23, 2218)])]
    sets2 += [(f"random_pair_{k}", [(22, rng.randrange(sm.n_inter)), (23, rng.randrange(sm.n_inter))]) for k in range(2)]
    for role, ns in sets2:
        conds.append(("n_sup+1", ns, role, neu, "softener test L22/L23 zero"))
    with open(args.out, "a") as f:
        for state, ns, role, ps, exp in conds:
            edits = [(l, n, "zero", 0.0) for l, n in ns]
            res = generate(sm, ps, state, edits=edits, batch=48)
            desc = describe(state, edits, unit="act")
            ivs = desc["intervention"] if isinstance(desc["intervention"], list) else [desc["intervention"]]
            for iv in ivs:
                iv["role"] = role
                iv["experiment"] = exp
            for p, (text, nt, fin) in zip(ps, res):
                f.write(json.dumps({"prompt_id": p["id"], "state": state, "intervention": desc["intervention"],
                                    "text": text, "n_tokens": nt, "finish": fin, "seed": 0,
                                    "decoding": {"do_sample": False, "max_new_tokens": 160}}) + "\n")
            f.flush()
            print(state, role, [f"L{l}/N{n}" for l, n in ns], len(ps), flush=True)


if __name__ == "__main__":
    main()
