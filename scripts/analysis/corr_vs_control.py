"""Amplifier/suppressor token correlation vs the neutral control's relative change, among the 1% of neurons most
changed by both poles of each trait.  python scripts/analysis/corr_vs_control.py METRICS.npz OUT.json"""
import json
import sys

import numpy as np

m = np.load(sys.argv[1])
st = list(m["states"])
rms0 = m["rms_a"][0]
floor = np.array([0.25 * np.median(r[r > 1e-3]) for r in rms0])[:, None]
den = np.maximum(rms0, floor)
zr = m["rms_d"] / den[None]
c = st.index("ctrl+1")
act = rms0 > 1e-3
out = {}
for t, (A, P) in {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}.items():
    a, p = st.index(A), st.index(P)
    corr = m[f"corr_{t}"]
    both = np.minimum(zr[a], zr[p])
    sel = act & (both > np.percentile(both[act], 99))
    r = zr[c] / both
    rows = []
    for lo, hi in zip([0, 0.25, 0.4, 0.55, 0.7], [0.25, 0.4, 0.55, 0.7, 1e9]):
        s = sel & (r >= lo) & (r < hi)
        rows.append({"lo": lo, "hi": hi, "n": int(s.sum()), "mean_corr": float(corr[s].mean()) if s.any() else None,
                     "frac_neg": float((corr[s] < 0).mean()) if s.any() else None})
        if s.any():
            print(f"{t:12s} control/trait [{lo:.2f},{hi:.2f}) n={s.sum():5d} mean corr {corr[s].mean():+.2f} frac<0 {(corr[s] < 0).mean():.2f}")
    rho = float(np.corrcoef(np.argsort(np.argsort(r[sel])), np.argsort(np.argsort(corr[sel])))[0, 1])
    print(t, "spearman:", round(rho, 3))
    out[t] = {"bins": rows, "spearman": rho, "ratio": r[sel].astype(float).tolist(), "corr": corr[sel].astype(float).tolist()}
json.dump(out, open(sys.argv[2], "w"))
