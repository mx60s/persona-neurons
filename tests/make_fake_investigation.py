"""Synthetic automated-investigation run for testing the explorer's Live run page and investigation sections.

    python tests/make_fake_investigation.py --out /tmp/pn_fake_inv             # copy runs/fake, add a finished run
    python tests/make_fake_investigation.py --out /tmp/pn_fake_inv --live      # then keep appending a neuron every 2 s
    python explorer/server.py --run /tmp/pn_fake_inv --no-model --port 8796

Copies runs/fake to --out (replacing an earlier copy made by this script), adds sample rows to neurons/index.json,
investigations/{progress.json, finished.jsonl, <id>.json}, investigation versions and tests on cards, and
population_stats.json. Refuses to write anywhere inside runs/ except a fresh copy. Everything is fake.
"""
import argparse
import datetime
import json
import os
import random
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "runs", "fake")
MARK = ".fake_investigation"

NICKS = [
    ("the \"so ___\" neuron", "Fires on the word after a degree \"so\" (\"so much\", \"so difficult\")."),
    ("the worry-verb neuron", "Fires on verbs of worrying and fearing (\"worry\", \"dread\", \"fear\")."),
    ("the \"what if\" neuron", "Fires on \"if\" after \"what\" when a sentence imagines something going wrong."),
    ("the apology neuron", "Fires on \"sorry\" and \"apologize\" at the start of a reply."),
    ("the curiosity neuron", "Fires on words that invite exploring an idea (\"imagine\", \"wonder\", \"explore\")."),
    ("the metaphor neuron", "Fires inside figurative comparisons, on the word after \"like\" or \"as if\"."),
    ("the comma-before-but neuron", "Fires on a comma that comes right before \"but\"."),
    ("the deadline neuron", "Fires on words about time pressure (\"deadline\", \"due\", \"late\")."),
    ("the novelty neuron", "Fires on \"new\", \"novel\" and \"unusual\" when they describe ideas, not objects."),
    ("the reassurance neuron", "Fires on phrases that calm someone down (\"it's okay\", \"don't worry\")."),
    ("the plural-s neuron", "Fires on a plural \"s\" after a noun about people."),
    ("the question-mark neuron", "Fires on question marks that end a worried question."),
    ("the art-words neuron", "Fires on words for art forms (\"painting\", \"poetry\", \"jazz\")."),
    ("the heartbeat neuron", "Fires on body words linked to anxiety (\"heart\", \"chest\", \"breath\")."),
]
TEXTS = [
    "I was so worried about the exam.", "The bus was late again.", "What if the plane crashes?",
    "She painted the sky in wild colors.", "My heart was pounding before the interview.", "The report is due tomorrow.",
    "Imagine a city built on water.", "He sat down at the table.", "Sorry, I didn't mean to interrupt.",
    "It's okay, take your time.", "The cat slept on the sofa.", "Is something wrong with me?",
]
STATUS_KEY = {"tested": "held_up", "mixed": "partly", "rejected": "not_held", "untested": "no_coherent"}


def now(dt=None):
    return (dt or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def jdump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def fname(nid):
    layer, n = nid[1:].split("/N")
    return f"L{layer}_N{n}"


def make_tests(rng, rnd, good):
    tests = []
    for k in range(rng.choice([4, 6])):
        exp = rng.choice(["high", "high", "low"])
        thr = round(rng.uniform(0.2, 0.6), 3)
        ok = rng.random() < good
        if exp == "high":
            val = round(thr * (rng.uniform(1.05, 2.5) if ok else rng.uniform(0.05, 0.9)), 3)
        else:
            val = round(thr * (rng.uniform(0.0, 0.9) if ok else rng.uniform(1.1, 2.0)), 3)
        t = rng.choice(TEXTS)
        tests.append({"test_id": f"r{rnd}t{k + 1}", "text": t, "expected": exp,
                      "target": t.split()[rng.randrange(len(t.split()))].strip(".,?") if exp == "high" else "",
                      "value": val, "threshold": thr, "pass": ok, "measured_max": round(max(val, 0.01), 3),
                      "measured_at": "target" if exp == "high" else "token"})
    return tests


def make_one(rng, nid, kind, trait, t_done):
    nick, one = rng.choice(NICKS)
    if kind == "random":
        status = rng.choices(["tested", "mixed", "rejected", "untested"], [0.12, 0.2, 0.43, 0.25])[0]
    else:
        status = rng.choices(["tested", "mixed", "rejected", "untested"], [0.38, 0.24, 0.26, 0.12])[0]
    good = {"tested": 0.92, "mixed": 0.65, "rejected": 0.3, "untested": 0.2}[status]
    n_rounds = rng.randint(1, 3)
    rounds, cost = [], 0.0
    for rnd in range(1, n_rounds + 1):
        last = rnd == n_rounds
        tests = make_tests(rng, rnd, good if last else good * 0.7)
        c = round(rng.uniform(0.002, 0.006), 4)
        cost += c
        rounds.append({
            "round": rnd,
            "hypothesis": {"nickname": nick if last else f"{nick} (draft {rnd})",
                           "one_liner": one if last else "Fires on " + rng.choice(["emotional words", "short function words", "the start of replies"]) + " (earlier guess).",
                           "trigger_hypothesis": one, "confidence": rng.choice(["low", "medium"])},
            "tests": tests, "passes": sum(t["pass"] for t in tests), "n_tests": len(tests),
            "decision": ("stop: no coherent explanation" if status == "untested" else "accept") if last else "revise",
            "note": "" if last else "Two predictions failed on sentences without the trigger word; narrowing.",
            "cost_usd": c,
        })
    final = rounds[-1]
    if status == "untested":
        nick, one = "no clear pattern", "The investigator could not find a pattern that predicted new sentences."
    rec = {"id": nid, "kind": kind, "trait": trait if kind == "trait" else None, "nickname": nick, "one_liner": one,
           "status": status, "passes": final["passes"], "n_final_tests": final["n_tests"], "rounds": n_rounds,
           "cost_usd": round(cost, 4), "finished_at": now(t_done)}
    inv = dict(rec, model="cheap-model (fake)", started=now(t_done - datetime.timedelta(seconds=40)),
               rounds=rounds, n_rounds=n_rounds,
               raw=[f"(fake raw model output for round {r['round']})" for r in rounds])
    return rec, inv


def add_card(run, rec, inv):
    path = os.path.join(run, "neurons", "cards", fname(rec["id"]) + ".json")
    card = json.load(open(path)) if os.path.exists(path) else {"id": rec["id"], "status": "untested", "versions": [],
                                                                "tests": [], "disagreements": []}
    v = 1 + max([x.get("v", 0) for x in card["versions"]] or [0])
    card["versions"].append({"v": v, "author": inv["model"], "created": rec["finished_at"], "kind": "investigation",
                             "blind": True, "label": rec["nickname"], "trigger_hypothesis": rec["one_liner"],
                             "output_hypothesis": "", "alternatives": [], "adapter_changes": "", "supporting": [],
                             "contradicting": [], "proposed_tests": [], "confidence": "low",
                             "no_coherent_hypothesis": rec["status"] == "untested"})
    for r in inv["rounds"]:
        card["tests"].append({"test_id": f"inv-r{r['round']}", "rule": "investigation", "round": r["round"],
                              "description": f"round {r['round']} predictions", "prediction": r["hypothesis"]["one_liner"],
                              "texts": [{k: t[k] for k in ("text", "expected", "measured_max", "measured_at", "value",
                                                           "threshold", "pass")} for t in r["tests"]],
                              "result": "pass" if r["passes"] == r["n_tests"] else "fail" if r["passes"] < r["n_tests"] / 2 else "mixed",
                              "state": "base", "notes": "fake"})
    card["status"] = rec["status"]
    jdump(path, card)


def summarize(rows):
    def block(rs):
        d = {"done": len(rs), "held_up": 0, "partly": 0, "not_held": 0, "no_coherent": 0}
        for r in rs:
            d[STATUS_KEY[r["status"]]] += 1
        return d
    by_kind = {k: block([r for r in rows if r["kind"] == k]) for k in ("trait", "random")}
    by_trait = {t: block([r for r in rows if r["kind"] == "trait" and r["trait"] == t]) for t in ("neuroticism", "openness")}
    return by_kind, by_trait


def write_progress(run, started, rows, n_total, running, failed, stale=False):
    cost = round(sum(r["cost_usd"] for r in rows), 4)
    per = round(cost / len(rows), 4) if rows else None
    left = n_total - len(rows) - failed
    by_kind, by_trait = summarize(rows)
    upd = datetime.datetime.now(datetime.timezone.utc) - (datetime.timedelta(minutes=5) if stale else datetime.timedelta(0))
    jdump(os.path.join(run, "investigations", "progress.json"), {
        "started": now(started), "updated": now(upd), "n_total": n_total, "n_done": len(rows),
        "n_running": running if left > 0 else 0, "n_failed": failed, "cost_usd": cost, "cost_cap_usd": 25.0,
        "est_cost_per_neuron": per, "eta_minutes": round(left * 0.03, 1) if left > 0 else 0,
        "by_kind": by_kind, "by_trait": by_trait})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/pn_fake_inv")
    ap.add_argument("--n-total", type=int, default=120, help="neurons in the fake sample")
    ap.add_argument("--n-done", type=int, default=45, help="neurons finished before --live starts")
    ap.add_argument("--live", action="store_true", help="append a finished neuron every --every seconds")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--stale", action="store_true", help="write progress.json with an old 'updated' time")
    ap.add_argument("--no-stats", action="store_true", help="do not write population_stats.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=None,
                    help="neuron indices for new sample rows go up to this (default: the fake model's width, 192); "
                         "raise it for large --n-total load tests")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    runs = os.path.join(REPO, "runs")
    if out == SRC or any(out == os.path.join(runs, x) or out.startswith(os.path.join(runs, x) + os.sep)
                         for x in os.listdir(runs)):
        sys.exit(f"refusing to write into existing run directory {out}")
    if os.path.exists(out):
        if not os.path.exists(os.path.join(out, MARK)):
            sys.exit(f"{out} exists and was not made by this script; choose another --out")
        shutil.rmtree(out)
    shutil.copytree(SRC, out, ignore=shutil.ignore_patterns("edits"))
    open(os.path.join(out, MARK), "w").write("made by tests/make_fake_investigation.py\n")
    os.makedirs(os.path.join(out, "investigations"), exist_ok=True)

    rng = random.Random(a.seed)
    man = json.load(open(os.path.join(out, "manifest.json")))
    n_layers, inter = man["model"]["arch"]["n_layers"], a.max_n or man["model"]["arch"]["inter"]
    idx_path = os.path.join(out, "neurons", "index.json")
    index = json.load(open(idx_path))
    have = {r["id"] for r in index}
    if a.n_total > n_layers * inter - len(have):
        sys.exit(f"--n-total {a.n_total} exceeds the {n_layers * inter - len(have)} free neuron ids; pass a larger --max-n")
    templates = [r for r in index if r.get("effects")]

    # sample: a few existing neurons (so some investigated neurons have full data) plus new ones
    sample, taken = [], set()
    for r in index[:6]:
        kind = "random" if r.get("kind") == "random_control" else "trait"
        sample.append((r["id"], kind, (r.get("selected_for") or ["neuroticism"])[0] if kind == "trait" else None))
    while len(sample) < a.n_total:
        nid = f"L{rng.randrange(n_layers)}/N{rng.randrange(inter)}"
        if nid in have or nid in taken:
            continue
        taken.add(nid)
        kind = "random" if rng.random() < 0.3 else "trait"
        sample.append((nid, kind, rng.choice(["neuroticism", "openness"]) if kind == "trait" else None))
    # the lead adds rows for new neurons before the run starts; hold back a few to test rows appearing later
    late = {s[0] for s in sample[-5:]}
    for nid, kind, trait in sample:
        if nid in have or nid in late:
            continue
        t = rng.choice(templates)
        layer, n = nid[1:].split("/N")
        eff = {s: {k: (round(v * rng.uniform(0.2, 1.6), 4) if isinstance(v, (int, float)) else v) for k, v in e.items()}
               for s, e in t["effects"].items()}
        index.append({"id": nid, "layer": int(layer), "n": int(n), "kind": "sample", "trait": trait,
                      "sample": {"kind": kind, "trait": trait}, "selected_for": [trait] if trait else [],
                      "group": {}, "rank": {}, "selection_reason": f"investigation sample ({kind})",
                      "base": t.get("base"), "effects": eff, "card_status": "untested", "label": ""})
    jdump(idx_path, index)

    if not a.no_stats:
        jdump(os.path.join(out, "population_stats.json"), {
            "neuroticism": {"n": 56, "gain_like": 41, "new_response": 9, "reply_start": 12, "median_ratio": 1.18},
            "openness": {"n": 48, "gain_like": 30, "new_response": 11, "reply_start": 7, "median_ratio": 1.09},
            "random": {"n": 36, "gain_like": 6, "new_response": 2, "reply_start": 3, "median_ratio": 1.01},
            "notes": "Fake numbers. gain_like: the adapter scales the neuron's existing peaks without moving them; "
                     "new_response: fires on text it ignored before; reply_start: most of the change is at the start "
                     "of the reply."})

    started = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=12)
    fin_path = os.path.join(out, "investigations", "finished.jsonl")
    rows, failed = [], 0

    def finish(k):
        nid, kind, trait = sample[k]
        t_done = started + datetime.timedelta(seconds=15 * (k + 1)) if not a.live or k < a.n_done else \
            datetime.datetime.now(datetime.timezone.utc)
        rec, inv = make_one(rng, nid, kind, trait, t_done)
        jdump(os.path.join(out, "investigations", fname(nid) + ".json"), inv)
        add_card(out, rec, inv)
        with open(fin_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        rows.append(rec)

    for k in range(min(a.n_done, len(sample))):
        finish(k)
    failed = 2
    write_progress(out, started, rows, len(sample), 4, failed, stale=a.stale)
    print(f"wrote {out}: {len(rows)} of {len(sample)} finished", flush=True)
    if not a.live:
        return
    k = len(rows)
    while k < len(sample) - failed:
        time.sleep(a.every)
        if sample[k][0] in late:   # append the held-back index row just before its result
            index = json.load(open(idx_path))
            nid, kind, trait = sample[k]
            layer, n = nid[1:].split("/N")
            index.append({"id": nid, "layer": int(layer), "n": int(n), "kind": "sample", "trait": trait,
                          "sample": {"kind": kind, "trait": trait}, "effects": {}, "card_status": "untested"})
            jdump(idx_path, index)
        finish(k)
        k += 1
        write_progress(out, started, rows, len(sample), 4, failed)
        print(f"finished {rows[-1]['id']} ({len(rows)}/{len(sample)})", flush=True)
    write_progress(out, started, rows, len(sample), 0, failed)
    print("done", flush=True)


if __name__ == "__main__":
    main()
