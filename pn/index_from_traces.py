"""Index rows and free per-neuron measurements computed directly from traces (no all-neuron statistics needed).

    python -m pn.index_from_traces --traces /root/out/traces_pilot --neurons runs/v2_work/pilot.json \
        --corpus data/corpus.jsonl --scatter runs/v1/scatter.npz --out runs/v2_work/pilot_candidates.json \
        --stats runs/v2_work/population_stats.json

Per neuron and polarity state s (all positions except each document's first token):
  mean/rms change vs base, z = change / denominator (base RMS, floored at 0.01), document agreement, 90% document
  bootstrap interval of the mean change, token-level amp/sup correlation per trait, share of the squared change on
  chat-template positions (reply start), and a change type:
    offset   |mean change| >= 0.7 x RMS change          (a near-constant push, content-independent)
    gain     corr(base, state) >= 0.9 and not offset    (same response, turned up or down)
    reshaped otherwise                                   (responds differently, not just scaled)
  with the gain factor = least-squares slope of state on base activations.
"""
import argparse
import json

import numpy as np

from pn.collect import template_mask

POLES = {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}
SCATTER_KEY = {"o_amp+1": "openness_amp", "o_sup+1": "openness_sup", "n_amp+1": "neuroticism_amp",
               "n_sup+1": "neuroticism_sup", "ctrl+1": "control"}


def change_type(b, a):
    d = a - b
    rms = float(np.sqrt((d ** 2).mean()))
    if rms < 1e-9:
        return "none", 1.0, 1.0
    corr = float(np.corrcoef(b, a)[0, 1]) if b.std() > 0 and a.std() > 0 else 0.0
    slope = float((b * a).sum() / max((b * b).sum(), 1e-12))
    if abs(d.mean()) >= 0.7 * rms:
        return "offset", corr, slope
    if corr >= 0.9:
        return "gain", corr, slope
    return "reshaped", corr, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--scatter", default="runs/v1/scatter.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    args = ap.parse_args()
    meta = json.load(open(f"{args.traces}/meta.json"))
    S = meta["states"]
    act = np.load(f"{args.traces}/act.npy", mmap_mode="r")
    act = np.ascontiguousarray(np.transpose(act, (2, 0, 1)))  # [K, S, T] in RAM: one contiguous block per neuron
    docs = [json.loads(l) for l in open(args.corpus)]
    valid = np.concatenate([np.r_[False, np.ones(len(d["ids"]) - 1, bool)] for d in docs])
    tmpl = np.concatenate([np.array(template_mask(d)) for d in docs])
    doc_of = np.concatenate([np.full(len(d["ids"]), i) for i, d in enumerate(docs)])[valid]
    D = len(docs)
    doc_n = np.bincount(doc_of, minlength=D).astype(float)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(200)]).astype(float) * doc_n[None]
    sc = np.load(args.scatter)
    kidx = {(l, n): k for k, (l, n) in enumerate(meta["neurons"])}
    rows = []
    for x in json.load(open(args.neurons)):
        l, n = x["layer"], x["n"]
        k = kidx[(l, n)]
        A = {s: act[k, S.index(s)].astype(np.float32) for s in S}
        b = A["base"][valid]
        rms_b = float(np.sqrt((b ** 2).mean()))
        den = max(rms_b, 0.01)
        q = {f"p{p}": float(np.quantile(b, p / 100 if p >= 1 else p / 100)) for p in (1, 50, 99)}
        rec = {"id": f"L{l}/N{n}", "layer": l, "n": n,
               "kind": "candidate" if x["kind"] == "trait" else "random_control",
               "selected_for": list(x.get("traits") or ([x["trait"]] if x.get("trait") else [])),
               "group": {t: "trait_specific" for t in (x.get("traits") or ([x["trait"]] if x.get("trait") else []))},
               "rank": {}, "selection_reason": ("random sample from the trait-specific population (change >= 0.5 x usual "
                                                "activity under a pole, neutral control < half as much)")
               if x["kind"] == "trait" else "layer-matched random comparison neuron",
               "base": {"mean": float(b.mean()), "rms": rms_b, "frac_pos": float((b > 0).mean()),
                        "frac_neg": float((b < 0).mean()), "max": float(b.max()), "min": float(b.min()),
                        "q": {"p01": float(np.quantile(b, 0.01)), "p50": float(np.quantile(b, 0.5)),
                              "p99": float(np.quantile(b, 0.99)), "p999": float(np.quantile(b, 0.999))}},
               "denominator": den, "effects": {}, "corr_amp_sup": {}, "layer_pct": {}, "template": {},
               "change_type": {}, "pilot": True}
        for s in S:
            if s == "base":
                continue
            a = A[s][valid]
            d = a - b
            dm = np.bincount(doc_of, weights=d, minlength=D) / np.maximum(doc_n, 1)
            bs = (W @ dm) / W.sum(1)
            md = float(d.mean())
            ag = float((np.sign(dm[doc_n > 0]) == np.sign(md)).mean())
            ct, corr, slope = change_type(b, a)
            dsq = (A[s] - A["base"]) ** 2
            tshare = float(dsq[valid & tmpl].sum() / max(dsq[valid].sum(), 1e-12))
            rec["effects"][s] = {"mean_diff": md, "rms_diff": float(np.sqrt((d ** 2).mean())), "z_mean": md / den,
                                 "z_rms": float(np.sqrt((d ** 2).mean())) / den, "agree": ag,
                                 "ci90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
                                 "template_share": tshare}
            if s in SCATTER_KEY:
                lay = sc[SCATTER_KEY[s]].astype(np.float32).reshape(36, -1)[l]
                rec["layer_pct"][s] = float((lay < lay[n]).mean())
                rec["change_type"][s] = {"type": ct, "corr_with_base": round(corr, 3), "gain": round(slope, 3)}
        for t, (pa, ps) in POLES.items():
            da, dp = A[pa][valid] - b, A[ps][valid] - b
            rec["corr_amp_sup"][t] = float(np.corrcoef(da, dp)[0, 1]) if da.std() > 0 and dp.std() > 0 else 0.0
            rec["template"][t] = {"template_share": max(rec["effects"][pa]["template_share"], rec["effects"][ps]["template_share"])}
        rec["template_driven"] = any(v["template_share"] > 0.5 for v in rec["template"].values())
        rows.append(rec)
    json.dump(rows, open(args.out, "w"), indent=1)

    # population summary: for each neuron, the change under the stronger pole of its trait (random: the stronger of all four)
    summ = {}
    for name, sel in (("neuroticism", lambda r: "neuroticism" in r["selected_for"]),
                      ("openness", lambda r: "openness" in r["selected_for"]),
                      ("random", lambda r: r["kind"] == "random_control")):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        types, ups, downs, opp, rstart = [], 0, 0, 0, 0
        for r in rs:
            poles = POLES[name] if name in POLES else ("o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1")
            s = max(poles, key=lambda p: r["effects"][p]["z_rms"])
            ct = r["change_type"][s]
            types.append(ct["type"])
            if ct["type"] == "gain":
                ups += ct["gain"] > 1
                downs += ct["gain"] <= 1
            if name in POLES:
                opp += r["corr_amp_sup"][name] < 0
            rstart += r["template_driven"]
        summ[name] = {"n": len(rs),
                      "same response, turned up or down (gain)": types.count("gain"),
                      "  of which turned up": ups, "  of which turned down": downs,
                      "near-constant push (offset)": types.count("offset"),
                      "reshaped / new response": types.count("reshaped"),
                      "mostly at the reply start": rstart}
        if name in POLES:
            summ[name]["amplifier and suppressor push opposite ways"] = opp
    summ["notes"] = ("Free measurements from the activations themselves, no AI explainer. Change type is judged under the "
                     "pole that changes the neuron most: 'gain' = activations stay tightly correlated with the original "
                     "(r >= 0.9) and are scaled; 'offset' = mostly a constant push; 'reshaped' = it responds differently. "
                     "Pilot sample: random draw from the trait-specific populations, plus layer-matched random neurons.")
    json.dump(summ, open(args.stats, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
