"""Second scoring rule for card tests, tied to each neuron's natural activation distribution (no GPU needed).

    python -m pn.rescore --run runs/v1

Rule v1 (pn.explain.score_text) asked a "high" text to reach half the median of the neuron's five strongest corpus
activations (roughly its 99.99th percentile) and a "low" text to stay near zero on every token. That rejected texts
that reached the neuron's 0.1st percentile, and failed "low" texts on unrelated tokens. Rule v2, kept alongside v1:
  high:     target value >= base p99 of the neuron on the corpus
  negative: target value <= base p01
  low:      does not cross the threshold of its partner's direction anywhere (max < p99 if the test has a high
            text, min > p01 if it has a negative text)
  test:     pass if every text passes; fail if the partner contrast goes the wrong way (high value <= low text max,
            or negative value >= low text min) or fewer than half the texts pass; else mixed
  card:     tested if >= 6 of 10 tests pass, mixed if 3 to 5, rejected if <= 2.
Random-control neurons get the same rule and act as a check on leniency.
"""
import argparse
import json
import os
from datetime import datetime, timezone


def rescore_test(t, q):
    p99, p01 = q["p99"], q["p01"]
    texts = t["texts"]
    has_high = any(x["expected"] == "high" for x in texts)
    has_neg = any(x["expected"] == "negative" for x in texts)
    recs = []
    for x in texts:
        if x["expected"] == "high":
            ok = x["value"] >= p99
        elif x["expected"] == "negative":
            ok = x["value"] <= p01
        else:
            ok = (not has_high or x["measured_max"] < p99) and (not has_neg or x["measured_min"] > p01)
        recs.append(ok)
    frac = sum(recs) / len(recs)
    wrong = False
    lows = [x for x in texts if x["expected"] == "low"]
    for x in texts:
        if x["expected"] == "high" and lows and x["value"] <= max(l["measured_max"] for l in lows):
            wrong = True
        if x["expected"] == "negative" and lows and x["value"] >= min(l["measured_min"] for l in lows):
            wrong = True
    if all(recs):
        return "pass", recs
    if wrong or frac < 0.5:
        return "fail", recs
    return "mixed", recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cdir = os.path.join(args.run, "neurons", "cards")
    ip = os.path.join(args.run, "neurons", "index.json")
    index = json.load(open(ip))
    kind = {r["id"]: r["kind"] for r in index}
    summary = {"candidate": [], "random_control": []}
    for f in sorted(os.listdir(cdir)):
        c = json.load(open(os.path.join(cdir, f)))
        if not c.get("tests"):
            continue
        q = json.load(open(os.path.join(args.run, "neurons", "detail", f)))["quantiles_base"]
        v1 = [t for t in c["tests"] if t.get("rule", "v1") == "v1"]
        latest_v = max(t["card_version"] for t in v1)
        v1 = [t for t in v1 if t["card_version"] == latest_v]
        c["tests"] = [t for t in c["tests"] if t.get("rule") != "v2"]
        res = []
        for t in v1:
            t.setdefault("rule", "v1")
            r, recs = rescore_test(t, q)
            nt = dict(t)
            nt.update({"rule": "v2", "result": r, "texts": [dict(x, pass_v2=ok) for x, ok in zip(t["texts"], recs)],
                       "thresholds_v2": {"p99": q["p99"], "p01": q["p01"]},
                       "notes": t.get("notes", "") + "; rescored with rule v2 (pn/rescore.py)",
                       "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            res.append(r)
            c["tests"].append(nt)
        npass = res.count("pass")
        old = c["status"]
        c["status"] = "tested" if npass >= 6 else "mixed" if npass >= 3 else "rejected"
        c.setdefault("status_history", []).append({"rule": "v1", "status": old})
        c["status_history"].append({"rule": "v2", "status": c["status"], "pass": npass, "of": len(res)})
        json.dump(c, open(os.path.join(cdir, f), "w"), indent=1)
        nid = c["id"]
        summary[kind.get(nid, "candidate")].append((nid, npass, c["status"]))
    for r in index:
        cp = os.path.join(cdir, r["id"].replace("/", "_") + ".json")
        if os.path.exists(cp):
            r["card_status"] = json.load(open(cp))["status"]
    json.dump(index, open(ip, "w"), indent=1)
    for k, v in summary.items():
        if not v:
            continue
        ps = [x[1] for x in v]
        st = [x[2] for x in v]
        print(f"{k}: n={len(v)} mean passes {sum(ps) / len(ps):.2f}/10; tested {st.count('tested')}, mixed {st.count('mixed')}, rejected {st.count('rejected')}")
        print("   ", sorted(v, key=lambda x: -x[1])[:12])


if __name__ == "__main__":
    main()
