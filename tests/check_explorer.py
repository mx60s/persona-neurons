"""Hits every explorer endpoint against a running server and checks the responses.

    python explorer/server.py --run runs/fake --no-model --port 8792 &
    python explorer/server.py --run runs/fake --model ../work/tiny/base --adapters ../work/tiny/adapters --port 8793 &
    python tests/check_explorer.py --port 8792 --no-model
    python tests/check_explorer.py --port 8793

Writes test entries to runs/<id>/edits/ (a card edit and saved results); remove edits/ afterwards if unwanted.
"""
import argparse
import json
import sys
import time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8791)
ap.add_argument("--no-model", action="store_true")
args = ap.parse_args()
B = f"http://127.0.0.1:{args.port}"
c = httpx.Client(base_url=B, timeout=300)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, detail)


def sse(path, body):
    evs = []
    with c.stream("POST", path, json=body) as r:
        if r.status_code != 200:
            return r.status_code, [json.loads(r.read())]
        for line in r.iter_lines():
            if line.startswith("data: "):
                evs.append(json.loads(line[6:]))
    return 200, evs


# static + local-only
r = c.get("/")
check("index.html served", r.status_code == 200 and "Persona neurons" in r.text)
for f in ("app.js", "style.css"):
    check(f"{f} served", c.get(f"/{f}").status_code == 200)
check("foreign Host rejected", c.get("/api/meta", headers={"host": "evil.example:80"}).status_code == 403)
check("cross-site origin rejected", c.get("/api/meta", headers={"origin": "http://evil.example"}).status_code == 403)

meta = c.get("/api/meta").json()
check("meta has manifest", meta["run"]["run_id"] and meta["run"]["traits"])
if not args.no_model:
    for _ in range(120):
        if meta["live"]["loaded"] or meta["live"]["error"]:
            break
        time.sleep(1)
        meta = c.get("/api/meta").json()
check("live status matches mode", meta["live"]["loaded"] == (not args.no_model), meta["live"]["status"])

rows = c.get("/api/neurons").json()
check("neurons list", len(rows) >= 15 and all("effects" in r for r in rows), f"n={len(rows)}")
groups = {g for r in rows for g in (r["group"] or {}).values() if g}
check("groups present", {"shared", "amp_primary", "sup_primary", "cross_trait"} <= groups, sorted(groups))
statuses = {r["card_status"] for r in rows}
check("all card statuses present", {"untested", "tested", "mixed", "rejected"} <= statuses, sorted(statuses))

nid = next(r["id"] for r in rows if r.get("has_profile"))
d = c.get("/api/neuron", params={"id": nid}).json()
check("neuron detail", d["detail"] and d["detail"]["examples"] and d["detail"]["trajectory"])
check("neuron card + profile", d["card"] and d["card"]["versions"] and d["profile"])
check("bad id rejected", c.get("/api/neuron", params={"id": "../../etc"}).status_code == 400)
for r in rows:
    dd = c.get("/api/neuron", params={"id": r["id"]})
    if dd.status_code != 200:
        check(f"neuron {r['id']}", False, dd.text)

# card edit: appended, original untouched
before = c.get("/api/card", params={"id": nid}).json()
n_before = len(before["versions"])
ed = c.post("/api/card", json={"id": nid, "label": "edited label", "status": "mixed", "alternatives": "a\nb",
                               "supporting": "x:e00, x:e01", "note": "test edit"}).json()
check("card edit appended", len(ed["versions"]) == n_before + 1 and ed["versions"][0]["kind"] == "human_edit"
      and ed["status"] == "mixed" and ed["versions"][0]["supporting"] == ["x:e00", "x:e01"])
check("bad status rejected", c.post("/api/card", json={"id": nid, "status": "proven"}).status_code == 400)
rows2 = c.get("/api/neurons").json()
check("list reflects edited status", next(r for r in rows2 if r["id"] == nid)["card_status"] == "mixed")

# save
s = c.post("/api/save", json={"neuron": nid, "kind": "chat", "state": "o_amp+1", "intervention": None, "input": "hi", "output": "x"})
check("save ok", s.status_code == 200)
check("save requires state+intervention", c.post("/api/save", json={"neuron": nid, "kind": "x"}).status_code == 400)
check("saved listed", c.get("/api/saved", params={"neuron": nid}).json()[0]["state"] == "o_amp+1")

# live endpoints
L, N = nid[1:].split("/N")
live_calls = [
    ("GET", "/api/live/calibration", {"neuron": nid, "state": "o_amp+1"}),
    ("POST", "/api/live/read", {"neuron": nid, "text": "I keep worrying about the results. 🙂", "states": ["base", "o_amp+1", "o_sup+1", "ctrl+1"]}),
    ("POST", "/api/live/next", {"neuron": nid, "state": "n_amp+1", "mode": "add", "value": 2, "text": "How are you?"}),
    ("POST", "/api/live/next", {"neuron": nid, "state": "base", "mode": "zero", "text": "How are you?", "prefix": "I"}),
    ("POST", "/api/live/next", {"neuron": nid, "state": "o_amp-0.5|ctrl+1", "mode": "mul", "value": 3, "text": "hi", "chat": False}),
    ("GET", "/api/live/lens", {"neuron": nid, "state": "o_sup+1"}),
    ("GET", "/api/live/connections", {"neuron": nid, "state": "n_sup+1"}),
]
for method, path, body in live_calls:
    r = c.get(path, params=body) if method == "GET" else c.post(path, json=body)
    if args.no_model:
        check(f"{path} 503 without model", r.status_code == 503 and "live model not loaded" in r.json()["error"])
    else:
        check(f"{path} ok", r.status_code == 200, r.text[:200] if r.status_code != 200 else "")
        j = r.json()
        if path.endswith("read"):
            check("read: one row per state, aligned", all(len(v) == len(j["tokens"]) for v in j["acts"].values())
                  and set(j["acts"]) == {"base", "o_amp+1", "o_sup+1", "ctrl+1"} and j["intervention"] is None)
            check("read: adapter changes activation", j["acts"]["base"] != j["acts"]["o_amp+1"])
        if path.endswith("next"):
            check("next: variants carry state + intervention", j["state"] and all(v["intervention"]["neuron"] == nid for v in j["variants"]),
                  f"KL={[v['kl'] for v in j['variants']]}")
        if path.endswith("calibration"):
            check("calibration fixed to base", j["reference"].startswith("base") and abs(j["unit_act"] - j["resid_p50"] / j["down_norm_base"]) < 1e-6)
            cal_base = c.get(path, params={"neuron": nid, "state": "base"}).json()
            check("calibration unit identical across states", abs(cal_base["unit_act"] - j["unit_act"]) < 1e-9)
        if path.endswith("lens"):
            other = c.get(path, params={"neuron": nid, "state": "base"}).json()
            check("lens is per state", j["state"] == "o_sup+1" and other["state"] == "base")
check("unknown adapter rejected" if not args.no_model else "unknown adapter (no model) 503",
      c.post("/api/live/read", json={"neuron": nid, "text": "x", "states": ["zz_amp+1"]}).status_code == (503 if args.no_model else 400))

code, evs = sse("/api/live/chat", {"message": "Tell me about waiting.", "states": ["base", "o_amp+1", "n_sup+1"], "max_new": 12})
if args.no_model:
    check("chat 503 without model", code == 503)
else:
    cols = evs[0]["columns"]
    texts = ["" for _ in cols]
    for e in evs:
        if "t" in e:
            texts[e["col"]] += e["t"]
    check("matched chat streams 3 columns", len(cols) == 3 and all(texts) and evs[-1].get("end") and not any("error" in e for e in evs),
          [t[:30] for t in texts])
    check("matched chat columns carry state, no intervention", [c_["state"] for c_ in cols] == ["base", "o_amp+1", "n_sup+1"]
          and all(c_["intervention"] is None for c_ in cols))

for mode, value, ncol in (("add", 3, 3), ("mul", 2, 3), ("zero", 0, 2)):
    code, evs = sse("/api/live/steer", {"neuron": nid, "state": "o_amp+1", "mode": mode, "value": value, "message": "Hi", "max_new": 10})
    if args.no_model:
        check(f"steer {mode} 503 without model", code == 503)
        continue
    head = evs[0]
    texts = ["" for _ in head["columns"]]
    for e in evs:
        if "t" in e:
            texts[e["col"]] += e["t"]
    ivs = [col["intervention"] for col in head["columns"]]
    check(f"steer {mode}: {ncol} columns, middle unmodified, state carried",
          len(ivs) == ncol and ivs[1] is None and all(col["state"] == "o_amp+1" for col in head["columns"])
          and not any("error" in e for e in evs), [t[:24] for t in texts])
    if mode == "add":
        check("steer add: + and - columns differ from unmodified", texts[0] != texts[1] or texts[2] != texts[1])

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
