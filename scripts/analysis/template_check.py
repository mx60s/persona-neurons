"""How much of each candidate's adapter-induced change sits on chat-template tokens (role tokens, special tokens,
the assistant header and empty think block)? Recomputes amp/sup correlations and effect sizes without them."""
import json
import sys

import numpy as np

corpus, traces, index_path = sys.argv[1:4]
docs = [json.loads(l) for l in open(corpus)]
meta = json.load(open(f"{traces}/meta.json"))
S = meta["states"]
act = np.load(f"{traces}/act.npy", mmap_mode="r")


def template_mask(d):
    ids = d["ids"]
    m = np.zeros(len(ids), bool)
    for i, t in enumerate(ids):
        if t >= 151643:
            m[i] = True
        if i > 0 and ids[i - 1] == 151644:
            m[i] = True
        if i > 1 and ids[i - 2] == 151644:
            m[i] = True
        b = d.get("boundary")
        if b is not None and d["domain"] in ("chat", "emotional_chat") and b - 6 <= i < b:
            m[i] = True
    return m


tmpl = np.concatenate([template_mask(d) for d in docs])
valid = np.concatenate([np.r_[False, np.ones(len(d["ids"]) - 1, bool)] for d in docs])
print(f"template tokens: {tmpl[valid].mean():.4f} of tokens")
index = json.load(open(index_path))
kidx = {f"L{l}/N{n}": k for k, (l, n) in enumerate(meta["neurons"])}
out = []
print(f"{'id':12s} {'group':16s} tmpl_share corr_all corr_no_tmpl  zA all->no  zP all->no")
for r in index:
    if r["kind"] != "candidate":
        continue
    k = kidx[r["id"]]
    b = act[S.index("base"), :, k].astype(np.float32)
    for t, (A, P) in {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}.items():
        if t not in r["selected_for"]:
            continue
        da = act[S.index(A), :, k].astype(np.float32) - b
        dp = act[S.index(P), :, k].astype(np.float32) - b
        m2 = valid & ~tmpl
        sq = da ** 2 + dp ** 2
        share = float(sq[valid & tmpl].sum() / sq[valid].sum())
        c1 = float(np.corrcoef(da[valid], dp[valid])[0, 1])
        c2 = float(np.corrcoef(da[m2], dp[m2])[0, 1])
        za = float(np.sqrt((da[m2] ** 2).mean()) / r["denominator"])
        zp = float(np.sqrt((dp[m2] ** 2).mean()) / r["denominator"])
        rec = {"id": r["id"], "trait": t, "group": r["group"].get(t), "template_share": share, "corr_all": c1,
               "corr_no_template": c2, "zA_all": r["effects"][A]["z_rms"], "zA_no_template": za,
               "zP_all": r["effects"][P]["z_rms"], "zP_no_template": zp}
        out.append(rec)
        print(f"{r['id']:12s} {t[:4] + ':' + str(rec['group']):16s} {share:9.2f} {c1:+8.2f} {c2:+11.2f}   "
              f"{rec['zA_all']:.2f}->{za:.2f}  {rec['zP_all']:.2f}->{zp:.2f}")
sh = np.array([x["template_share"] for x in out])
print("median template share of squared change:", round(float(np.median(sh)), 3), "; share > 0.5:", round(float((sh > 0.5).mean()), 3))
json.dump(out, open("runs/v1/template_check.json", "w"), indent=1)
