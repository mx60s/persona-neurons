"""Teacher-forced activation patching: how much of an adapter's effect on its own outputs runs through a neuron set?

    python -m pn.patching --run runs/v1 --responses runs/v1/behavior/responses.jsonl --out runs/v1/patching.json

For an adapted state A (e.g. o_amp+1) and its held-out responses on its own trait's prompts, with tokens fixed:
    LL_base  = log-likelihood of the response tokens in the base state
    LL_A     = in state A
    LL_A|S   = in state A with the neurons in set S replaced by their base-state activations at the same positions
    LL_b|S   = in the base state with the neurons in S set to their state-A activations (the reverse direction)
    removed  = (LL_A - LL_A|S) / (LL_A - LL_base)     share of A's likelihood advantage removed by restoring S
    added    = (LL_b|S - LL_base) / (LL_A - LL_base)  share recovered by installing A's values of S in base
Sets: each trait's candidate groups, all of the trait's candidates, the cross-trait/generic group, single detailed
neurons, and layer-matched random sets of the same size (several draws). Patching acts on the down_proj input, so
it removes the change in the neuron's activation; any change in the neuron's own down column under A is kept.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from pn.measure import Measurer

TRAIT_STATES = {"openness": ["o_amp+1", "o_sup+1"], "neuroticism": ["n_amp+1", "n_sup+1"]}


class Patcher:
    def __init__(self, sm):
        self.sm = sm

    def _ids(self, prompt, response):
        p = self.sm.chat_ids(prompt)
        r = self.sm.tok(response, add_special_tokens=False).input_ids + [self.sm.tok.convert_tokens_to_ids("<|im_end|>")]
        return p, r

    def run(self, ids, nprompt, state, record=None, replace=None):
        """Sum log-prob of the response tokens. record: {layer: idx tensor} to capture; replace: {layer: (idx, values)}."""
        sm = self.sm
        sm.set_state(state)
        caps, hs = {}, []
        for layer in set(record or {}) | set(replace or {}):
            def hook(mod, inp, layer=layer):
                h = inp[0]
                if record and layer in record:
                    caps[layer] = h[0, :, record[layer]].clone()
                if replace and layer in replace:
                    idx, vals = replace[layer]
                    h = h.clone()
                    h[0, :, idx] = vals.to(h.dtype)
                    return (h,)
            hs.append(sm.down_proj_module(layer).register_forward_pre_hook(hook))
        try:
            with torch.inference_mode():
                x = torch.tensor([ids], device=sm.device)
                logits = sm.model(input_ids=x, use_cache=False).logits[0, nprompt - 1:-1].float()
                lp = torch.log_softmax(logits, -1).gather(1, x[0, nprompt:, None])[:, 0]
        finally:
            for h in hs:
                h.remove()
        return float(lp.sum()), caps

    def evaluate(self, prompt, response, state, sets):
        p, r = self._ids(prompt, response)
        ids, npr = p + r, len(p)
        allL = {}
        for name, neurons in sets.items():
            for l, n in neurons:
                allL.setdefault(l, set()).add(n)
        rec = {l: torch.tensor(sorted(v), device=self.sm.device) for l, v in allL.items()}
        ll_b, cap_b = self.run(ids, npr, "base", record=rec)
        ll_a, cap_a = self.run(ids, npr, state, record=rec)
        out = {"ll_base": ll_b, "ll_state": ll_a, "n_tokens": len(r), "sets": {}}
        for name, neurons in sets.items():
            byl = {}
            for l, n in neurons:
                byl.setdefault(l, []).append(n)
            rep_a, rep_b = {}, {}
            for l, ns in byl.items():
                cols = [sorted(allL[l]).index(n) for n in ns]
                idx = torch.tensor(ns, device=self.sm.device)
                rep_a[l] = (idx, cap_b[l][:, cols])  # base values into state
                rep_b[l] = (idx, cap_a[l][:, cols])  # state values into base
            ll_as, _ = self.run(ids, npr, state, replace=rep_a)
            ll_bs, _ = self.run(ids, npr, "base", replace=rep_b)
            out["sets"][name] = {"ll_state_patched": ll_as, "ll_base_patched": ll_bs}
        return out


def random_matched(neurons, n_inter, exclude, rng):
    out = []
    for l, _ in neurons:
        while True:
            n = rng.randrange(n_inter)
            if (l, n) not in exclude and (l, n) not in out:
                break
        out.append((l, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--prompts", default="data/behavior_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--max-prompts", type=int, default=40)
    ap.add_argument("--singles", default="", help="comma-separated neuron ids to patch individually")
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.run, "neurons", "index.json")))
    prompts = {p["id"]: p for p in json.load(open(args.prompts))}
    resp = {}
    for l in open(args.responses):
        r = json.loads(l)
        if r.get("intervention") is None:
            resp[(r["prompt_id"], r["state"])] = r["text"]
    m = Measurer()
    sm = m.sm
    P = Patcher(sm)
    rng = random.Random(0)
    allc = {(r["layer"], r["n"]) for r in index}
    results = {}
    for trait, states in TRAIT_STATES.items():
        cands = [r for r in index if r["kind"] == "candidate" and trait in r["selected_for"]]
        sets = {f"{trait}:all ({len(cands)})": [(r["layer"], r["n"]) for r in cands]}
        for g in ("shared", "amp_primary", "sup_primary"):
            s = [(r["layer"], r["n"]) for r in cands if r["group"].get(trait) == g]
            if s:
                sets[f"{trait}:{g} ({len(s)})"] = s
        gen = [(r["layer"], r["n"]) for r in index if "cross" in r["selected_for"]]
        sets[f"cross_trait ({len(gen)})"] = gen
        for k in range(args.n_random):
            sets[f"random_matched_{k} ({len(cands)})"] = random_matched(sets[f"{trait}:all ({len(cands)})"], sm.n_inter, allc, rng)
        for nid in [x for x in args.singles.split(",") if x]:
            l, n = int(nid.split("/")[0][1:]), int(nid.split("/")[1][1:])
            sets[f"single {nid}"] = [(l, n)]
        pids = [p for p in prompts if prompts[p]["trait"] == trait][:args.max_prompts]
        for st in states:
            rows = []
            for pid in pids:
                if (pid, st) not in resp:
                    continue
                rows.append({"prompt_id": pid, **P.evaluate(prompts[pid]["text"], resp[(pid, st)], st, sets)})
            gap = np.array([r["ll_state"] - r["ll_base"] for r in rows])
            summ = {"n": len(rows), "mean_gap_nats": float(gap.mean()), "sets": {}}
            for name in sets:
                rem = np.array([r["ll_state"] - r["sets"][name]["ll_state_patched"] for r in rows])
                add = np.array([r["sets"][name]["ll_base_patched"] - r["ll_base"] for r in rows])
                bs = [rng.choices(range(len(rows)), k=len(rows)) for _ in range(1000)]
                frac_rem = rem.sum() / gap.sum()
                frac_add = add.sum() / gap.sum()
                ci = np.percentile([rem[b].sum() / gap[b].sum() for b in bs], [5, 95])
                ci_a = np.percentile([add[b].sum() / gap[b].sum() for b in bs], [5, 95])
                summ["sets"][name] = {"removed_frac": float(frac_rem), "removed_ci90": ci.tolist(),
                                      "added_frac": float(frac_add), "added_ci90": ci_a.tolist(),
                                      "neurons": [f"L{l}/N{n}" for l, n in sets[name]]}
                print(f"{st:9s} {name:32s} removed {frac_rem:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  added {frac_add:+.3f}", flush=True)
            results[st] = {"summary": summ, "rows": rows}
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
