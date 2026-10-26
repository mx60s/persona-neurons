"""Robustness of per-neuron effects to rare extreme tokens, from traces.

    python -m pn.robust --run runs/v1 --traces /workspace/out/traces_v1

For each traced neuron and polarity state: share of the summed squared change (state - base) contributed by the top
0.1% of tokens, the RMS change with those tokens removed (trimmed, in units of the neuron's denominator), the median
absolute change, and the token-level amp/sup correlation with and without the extreme tokens. Writes
`robust` into each neuron's index row and detail file.
"""
import argparse
import json
import os

import numpy as np

POLES = {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--top", type=float, default=0.001)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.traces, "meta.json")))
    S = meta["states"]
    act = np.load(os.path.join(args.traces, "act.npy"), mmap_mode="r")
    off, lens = np.array(meta["offsets"]), np.array(meta["lengths"])
    valid = np.ones(int(off[-1] + lens[-1]), bool)
    valid[off] = False
    kidx = {f"L{l}/N{n}": k for k, (l, n) in enumerate(meta["neurons"])}
    ip = os.path.join(args.run, "neurons", "index.json")
    index = json.load(open(ip))
    for rec in index:
        k = kidx.get(rec["id"])
        if k is None:
            continue
        base = act[S.index("base"), :, k].astype(np.float32)[valid]
        den = rec["denominator"]
        out = {}
        for st in ("o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", "ctrl+1"):
            d = act[S.index(st), :, k].astype(np.float32)[valid] - base
            sq = d * d
            ntop = max(1, int(args.top * len(d)))
            top = np.argpartition(-sq, ntop)[:ntop]
            keep = np.ones(len(d), bool)
            keep[top] = False
            out[st] = {"top_share": float(sq[top].sum() / max(sq.sum(), 1e-12)),
                       "z_rms_trimmed": float(np.sqrt(sq[keep].mean()) / den),
                       "z_mean_trimmed": float(d[keep].mean() / den),
                       "median_abs_z": float(np.median(np.abs(d)) / den)}
        for t, (a, p) in POLES.items():
            da = act[S.index(a), :, k].astype(np.float32)[valid] - base
            dp = act[S.index(p), :, k].astype(np.float32)[valid] - base
            ext = np.abs(da) + np.abs(dp)
            ntop = max(1, int(args.top * len(da)))
            keep = np.ones(len(da), bool)
            keep[np.argpartition(-ext, ntop)[:ntop]] = False
            out[f"corr_{t}"] = float(np.corrcoef(da, dp)[0, 1])
            out[f"corr_{t}_trimmed"] = float(np.corrcoef(da[keep], dp[keep])[0, 1])
        rec["robust"] = out
        dp_ = os.path.join(args.run, "neurons", "detail", rec["id"].replace("/", "_") + ".json")
        det = json.load(open(dp_))
        det["robust"] = out
        json.dump(det, open(dp_, "w"))
    json.dump(index, open(ip, "w"), indent=1)
    print(f"{'id':12s} {'group':28s} {'top0.1%share':>12s} {'z':>6s} {'z_trim':>6s} {'corr':>6s} {'corr_trim':>9s}")
    for rec in index:
        if "robust" not in rec or rec["kind"] != "candidate":
            continue
        t = next((x for x in ("openness", "neuroticism") if x in rec["selected_for"]), "openness")
        g = rec["group"].get(t) or next(iter(rec["group"].values()))
        a, p = POLES[t]
        st = a if rec["effects"][a]["z_rms"] >= rec["effects"][p]["z_rms"] else p
        r = rec["robust"]
        print(f"{rec['id']:12s} {t[:4] + ':' + g:28s} {r[st]['top_share']:12.2f} {rec['effects'][st]['z_rms']:6.2f} "
              f"{r[st]['z_rms_trimmed']:6.2f} {r['corr_' + t]:6.2f} {r['corr_' + t + '_trimmed']:9.2f}")


if __name__ == "__main__":
    main()
