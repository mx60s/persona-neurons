"""Summarize the replication runs against the unmodified states (paired by prompt)."""
import json
import random
import re

P = {p["id"]: p for p in json.load(open("runs/v1/behavior/prompts_all.json"))}
resp = {}
for l in open("runs/v1/behavior/responses.jsonl"):
    r = json.loads(l)
    if r.get("intervention") is None:
        resp[(r["prompt_id"], r["state"])] = r["text"]
base_sc = {}
for l in open("runs/v1/behavior/scores.jsonl"):
    r = json.loads(l)
    if r.get("intervention") is None:
        base_sc[(r["prompt_id"], r["state"], r["trait"])] = r["score"]
rep = [json.loads(l) for l in open("runs/v1/behavior/replication.jsonl")]
sc = {}
for l in open("runs/v1/behavior/replication/scores.jsonl"):
    r = json.loads(l)
    iv = r["intervention"]
    role = (iv[0] if isinstance(iv, list) else iv)["role"]
    sc[(r["prompt_id"], r["state"], role, r["trait"])] = r["score"]
SOFT = re.compile(r"\b(a bit|slightly|somewhat|a little|kind of|a touch|mildly)\b", re.I)
OH = re.compile(r"^\W*oh\b", re.I)
rng = random.Random(0)


def boot(d, n=2000):
    m = [sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n)]
    m.sort()
    return round(m[int(0.05 * n)], 2), round(m[int(0.95 * n)], 2)


out = {}
for r in rep:
    iv = r["intervention"]
    role = (iv[0] if isinstance(iv, list) else iv)["role"]
    out.setdefault((r["state"], role), []).append(r)
for (state, role), rows in out.items():
    sets = {"neuroticism": [x for x in rows if P[x["prompt_id"]]["trait"] == "neuroticism"],
            "general": [x for x in rows if P[x["prompt_id"]]["trait"] == "general"]}
    for name, rs in sets.items():
        if not rs:
            continue
        d = [sc[(x["prompt_id"], state, role, "neuroticism")] - base_sc[(x["prompt_id"], state, "neuroticism")]
             for x in rs if (x["prompt_id"], state, role, "neuroticism") in sc and (x["prompt_id"], state, "neuroticism") in base_sc]
        oh_i = sum(bool(OH.match(x["text"])) for x in rs)
        oh_u = sum(bool(OH.match(resp[(x["prompt_id"], state)])) for x in rs)
        so_i = sum(len(SOFT.findall(x["text"])) for x in rs)
        so_u = sum(len(SOFT.findall(resp[(x["prompt_id"], state)])) for x in rs)
        print(f"{state:8s} {role:16s} {name:11s} n={len(rs):3d} d_neuroticism={sum(d) / len(d):+.2f} {boot(d)}  "
              f"'Oh' openers {oh_u}->{oh_i}  softeners {so_u}->{so_i}")
