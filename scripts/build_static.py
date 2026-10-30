"""Build a static copy of the neuron browser (no server, no GPU), e.g. for GitHub Pages.

    python scripts/build_static.py --run runs/v2 --out site/
    python scripts/build_static.py --run runs/v2 --out site/ --intro data/plain_intro.md   # newer Start page text

Writes site/index.html, app.js, style.css, figures/*.png and data/:
  meta.json      what /api/meta returns, minus the live model and local paths
  plain.json     intro_md, glossary, adapters and the hand-written neuron entries (/api/plain?lite=1)
  results.json   exactly what /api/results returns
  list.json      one compact row per neuron: everything the list view, its facets and sorts need
  n/L_N.json     one trimmed /api/neuron response per neuron
All URLs in the page are relative, so the site works under any base path. The page runs in STATIC mode
(window.PN_STATIC): it reads these files and does list filtering, facet counts, sorting and paging in the browser.
Reads the run through explorer/server.py so the numbers match the live server; never writes to the run.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "explorer"))
sys.path.insert(0, ROOT)

# explorer/server.py imports pn.states (and torch through it) only for the live endpoints; stub it out
_stub = types.ModuleType("pn.states")
_stub.parse_state = lambda s: s
_stub.state_id = lambda s: s
sys.modules.setdefault("pn.states", _stub)
import server as SV  # noqa: E402

POLAR = ["o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", "ctrl+1"]
KEEP_STATES = ["base"] + POLAR
EFFECT_KEYS = ("mean_diff", "rms_diff", "z_mean", "z_rms", "agree", "resid_mean_diff")
# example windows kept per neuron: plain.json highlights first, then these kinds in order, up to MAX_EX
EX_QUOTA = [("top_pos", 3), ("top_neg", 2), ("top_diff", 3), ("quantile", 1), ("random", 1), ("training", 2),
            ("validation", 1)]
MAX_EX = 11
MAX_VERSIONS = 3


class StaticRun(SV.Run):
    """Read-only Run: skips creating edits/ in the run directory."""

    def __init__(self, path):  # noqa: D401 (same fields as Run, no makedirs)
        import collections
        import threading
        self.path = os.path.abspath(path)
        if not os.path.exists(os.path.join(self.path, "manifest.json")):
            raise SystemExit(f"{self.path} has no manifest.json")
        self._cache = collections.OrderedDict()
        self._small = {}
        self._finished = {"stamp": None, "offset": 0, "rows": [], "by_id": {}}
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()


def rnd(x, sig=4):
    """Round every float in a structure to `sig` significant digits."""
    if isinstance(x, float):
        if not math.isfinite(x) or x == 0:
            return x if math.isfinite(x) else None
        return float(f"{x:.{sig}g}")
    if isinstance(x, dict):
        return {k: rnd(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, sig) for v in x]
    return x


def r2(a):
    """Activations to 2 decimals, stored as integers x100 (the page divides by 100; detail["acts_scale"])."""
    return [None if v is None else int(round(v * 100)) for v in a]


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), ensure_ascii=False)
    return os.path.getsize(path)


# ---------------------------------------------------------------- list.json

def list_row(c):
    """Catalog record (server.build_catalog) -> compact row. Field names stay the server's so the JS port of
    list_rows() can use them directly; the search string is rebuilt in the browser from id, name, one and `sx`."""
    out = {"id": c["id"], "layer": c["layer"], "n": c["n"], "kind": c["kind"], "samples": sorted(c["samples"]),
           "traits": c["traits"], "name": c["name"], "name_src": c["name_src"], "one": c["one"], "one_src": c["one_src"],
           "verdict": c["verdict"], "effect": rnd(float(c["effect"] or 0), 6),  # sort key: keep enough digits that ties match the server
           "arrows": [[s, d, None if r is None else round(r, 2)] for s, d, r in c["arrows"]]}
    for k in ("hand", "hand_src", "inv", "where"):
        if c[k] is not None:
            out[k] = c[k]
    ch = {k: rnd(v, 3) for k, v in c["change"].items() if v}
    if ch:
        out["change"] = ch
    op = {k: v for k, v in c["opp"].items() if v}
    if op:
        out["opp"] = op
    tech = {"z_mean": {s: rnd(float(v), 6) for s, v in c["tech"]["z_mean"].items() if v is not None}}
    for k in ("card_status", "detailed", "template_driven", "outlier"):
        if c["tech"].get(k) not in (None, False):
            tech[k] = c["tech"][k]
    out["tech"] = tech
    base = " ".join(str(x or "") for x in (c["id"], c["name"], c["one"])).lower()
    extra = [w for w in c["search"].split() if w not in base.split()]
    if extra:
        out["sx"] = " ".join(dict.fromkeys(extra))
    return out


# ---------------------------------------------------------------- per-neuron file

def pick_examples(examples, highlights):
    by_id = {e.get("ex_id"): e for e in examples}
    picked, seen = [], set()

    def add(e):
        if e is not None and e.get("ex_id") not in seen and len(picked) < MAX_EX:
            picked.append(e)
            seen.add(e.get("ex_id"))

    for h in highlights or []:
        add(by_id.get(h))
    for kind, n in EX_QUOTA:
        for e in [e for e in examples if e.get("kind") == kind and e.get("ex_id") not in seen][:n]:
            add(e)
    for e in examples:  # fill the rest in file order
        if e.get("kind") != "training":
            add(e)
    order = {e.get("ex_id"): i for i, e in enumerate(examples)}
    return sorted(picked, key=lambda e: order.get(e.get("ex_id"), 0))


def trim_example(e):
    out = {k: e[k] for k in ("ex_id", "doc_id", "domain", "source", "kind", "split", "tokens", "start", "focus",
                             "boundary", "train_meta") if e.get(k) is not None}
    out["acts"] = {s: r2(a) for s, a in (e.get("acts") or {}).items() if s in KEEP_STATES}
    for k in ("gate", "up"):
        if (e.get(k) or {}).get("base"):
            out[k] = {"base": r2(e[k]["base"])}
    return out


def trim_detail(d, highlights):
    if not d:
        return None
    keep = lambda m: {s: v for s, v in (m or {}).items() if s in KEEP_STATES}
    w = d.get("weights") or {}
    out = {"id": d.get("id"), "trajectory": rnd(d.get("trajectory") or {}),
           "by_domain": rnd(keep(d.get("by_domain"))), "by_segment": rnd(keep(d.get("by_segment"))),
           "weights": {k: rnd(keep(v), 7) for k, v in w.items() if isinstance(v, dict)},
           "lens": keep(d.get("lens")), "natural_range": rnd(d.get("natural_range")),
           "examples": [trim_example(e) for e in pick_examples(d.get("examples") or [], highlights)],
           "n_examples_total": len(d.get("examples") or []), "acts_scale": 100}
    for k in ("steering", "ablation"):
        if d.get(k):
            out[k] = rnd(d[k])
    for t in out["trajectory"].values():
        for p in t:
            p.pop("label", None)
    return out


def trim_index(row):
    if not row:
        return None
    r = {k: v for k, v in row.items() if k not in ("pilot", "effects")}
    r["effects"] = {s: {k: e.get(k) for k in EFFECT_KEYS if e.get(k) is not None}
                    for s, e in (row.get("effects") or {}).items()}
    return rnd(r)


VERSION_KEYS = ("v", "kind", "author", "created", "blind", "confidence", "status_set", "label", "nickname",
                "one_liner", "no_coherent_hypothesis", "trigger_hypothesis", "output_hypothesis", "alternatives",
                "adapter_changes", "supporting", "contradicting", "proposed_tests", "note", "usage", "source")
TEXT_KEYS = ("text", "expected", "measured_max", "measured_at")


def trim_card(card):
    """Latest versions and test tables. The investigator's per-round tests are kept only for its final round
    (the other rounds are in the investigation section)."""
    if not card:
        return None
    vers = []
    for v in card.get("versions", [])[:MAX_VERSIONS]:
        x = {k: v[k] for k in VERSION_KEYS if v.get(k) not in (None, "", [])}
        if isinstance(x.get("usage"), dict):
            x["usage"] = {k: x["usage"].get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cost_usd")}
        vers.append(x)
    tests = []
    for t in card.get("tests", []):
        if t.get("rule") == "investigation" and not t.get("final"):
            continue
        x = {k: t[k] for k in ("test_id", "description", "prediction", "result", "state", "notes", "source")
             if t.get(k) is not None}
        x["texts"] = [{k: tx[k] for k in TEXT_KEYS if tx.get(k) not in (None, "")} for tx in t.get("texts") or []]
        tests.append(x)
    return rnd({"id": card.get("id"), "status": card.get("status"), "status_from": card.get("status_from"),
                "run_status": card.get("run_status"), "versions": vers, "tests": tests,
                "n_versions_total": len(card.get("versions", [])),
                "disagreements": card.get("disagreements", [])})


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def threshold_num(t):
    if isinstance(t, (int, float)):
        return t
    m = NUM_RE.search(str(t or ""))
    return float(m.group()) if m else None


def trim_investigation(I):
    """Rounds with their hypothesis and measured test table; no raw model outputs or prompts."""
    if not I:
        return None
    rounds = []
    for rd in I.get("rounds") or []:
        h = rd.get("hypothesis") or {}
        x = {"round": rd.get("round"), "decision": h.get("decision") if isinstance(h, dict) else None,
             "cost_usd": (rd.get("usage") or {}).get("cost_usd")}
        if isinstance(h, dict):
            x["hypothesis"] = {k: h[k] for k in ("nickname", "label", "one_liner", "trigger_hypothesis", "confidence")
                               if h.get(k)}
        elif h:
            x["hypothesis"] = h
        res = rd.get("results") or rd.get("tests") or []
        x["tests"] = [{"text": t.get("text"), "expected": t.get("expected"), "target": t.get("target") or None,
                       "value": (t.get("base") or {}).get("value", t.get("value")),
                       "threshold": threshold_num(t.get("threshold")), "pass": t.get("pass")} for t in res]
        for t in x["tests"]:  # "low" thresholds are shown as a +/- band
            if t["expected"] == "low" and t["threshold"] is not None:
                t["threshold"] = abs(t["threshold"])
        if not rd.get("valid", True) and rd.get("errors"):
            x["note"] = "invalid output: " + "; ".join(map(str, rd["errors"]))[:300]
        rounds.append({k: v for k, v in x.items() if v is not None})
    keys = ("id", "kind", "trait", "model", "status", "card_status", "passes", "n_final_tests", "rounds_done",
            "finished_at")
    out = {k: I.get(k) for k in keys if I.get(k) is not None}
    out["cost_usd"] = (I.get("usage") or {}).get("cost_usd")
    out["rounds"] = rounds
    return rnd(out)


def neuron_file(nid):
    d = SV.neuron(nid)
    P = d.get("plain") or {}
    return {"id": nid, "index": trim_index(d["index"]), "detail": trim_detail(d["detail"], P.get("highlights")),
            "card": trim_card(d["card"]), "profile": d["profile"], "saved": [],
            "investigation": trim_investigation(d["investigation"]), "plain": rnd(d["plain"]) if d["plain"] else None,
            "posctrl": d["posctrl"]}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/v2")
    ap.add_argument("--out", default="site")
    ap.add_argument("--posctrl", default=None, help="positive-control run (default: <run>_posctrl if present)")
    ap.add_argument("--intro", default=None, help="markdown file to use as the Start page instead of plain.json "
                    "intro_md (e.g. data/plain_intro.md, without re-running pn/plain.py)")
    ap.add_argument("--limit", type=int, default=0, help="only write this many neuron files (for testing)")
    args = ap.parse_args()

    SV.ARGS = argparse.Namespace(posctrl=args.posctrl, port=0, no_model=True)
    SV.RUN = StaticRun(args.run)
    SV.LIVE.update(status="static copy", error=None)
    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "data", "n"))

    # page: same files as the server's, with the STATIC flag set before app.js runs
    st = os.path.join(ROOT, "explorer", "static")
    html = open(os.path.join(st, "index.html")).read()
    flag = '<script>window.PN_STATIC = true;</script>\n'
    html = html.replace('<script type="module" src="app.js">', flag + '<script type="module" src="app.js">')
    assert "PN_STATIC" in html
    open(os.path.join(out, "index.html"), "w").write(html)
    for f in os.listdir(st):
        if f != "index.html" and os.path.isfile(os.path.join(st, f)):
            shutil.copy(os.path.join(st, f), out)
    open(os.path.join(out, ".nojekyll"), "w").close()

    figs = set()
    for d in (os.path.join(ROOT, "figures"), SV.RUN.p("figures")):
        for p in glob.glob(os.path.join(d, "*")):
            if SV.FIG_RE.match(os.path.basename(p)):
                os.makedirs(os.path.join(out, "figures"), exist_ok=True)
                shutil.copy(p, os.path.join(out, "figures"))
                figs.add(os.path.basename(p))

    sizes = {}
    meta = SV.meta()
    meta.pop("run_path", None)
    meta.update(live={"loaded": False, "status": "static copy", "error": None, "info": None},
                has_progress=False, static=True)
    sizes["meta.json"] = dump(meta, os.path.join(out, "data", "meta.json"))
    p = SV.plain(lite=1)
    plain = {"intro_md": p.get("intro_md"), "glossary": p.get("glossary"), "adapters": p.get("adapters"),
             "neurons": rnd(p.get("neurons") or {})} if isinstance(p, dict) else None
    if plain and args.intro:
        plain["intro_md"] = open(args.intro).read()
    if plain:
        sizes["plain.json"] = dump(plain, os.path.join(out, "data", "plain.json"))
    sizes["results.json"] = dump(SV.results(), os.path.join(out, "data", "results.json"))
    cat = SV.RUN.catalog()
    sizes["list.json"] = dump({"rows": [list_row(c) for c in cat]}, os.path.join(out, "data", "list.json"))

    ids = [c["id"] for c in cat]
    for nid, f in SV.RUN.finished()["by_id"].items():  # finished but missing from the index (none in v2)
        if nid not in set(ids):
            ids.append(nid)
    if args.limit:
        keep = [i for i in ids if os.path.exists(SV.RUN.p("profiles", SV.fid(i)[0] + ".md"))]
        ids = keep + [i for i in ids if i not in keep][:max(0, args.limit - len(keep))]
    nsz = []
    for k, nid in enumerate(ids):
        nsz.append(dump(neuron_file(nid), os.path.join(out, "data", "n", SV.fid(nid)[0] + ".json")))
        for rel in [r for r in SV.RUN._cache if r.startswith(("neurons/detail/", "neurons/cards/", "investigations/L"))]:
            SV.RUN._cache.pop(rel, None)  # big per-neuron files are read once
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(ids)} neuron files", flush=True)

    total = 0
    biggest = (0, "")
    for dp, _, fs in os.walk(out):
        for f in fs:
            s = os.path.getsize(os.path.join(dp, f))
            total += s
            biggest = max(biggest, (s, os.path.relpath(os.path.join(dp, f), out)))
    mb = lambda b: f"{b / 1e6:.2f} MB"
    print(f"site: {out}")
    for k, v in sizes.items():
        print(f"  data/{k:14s} {mb(v)}")
    print(f"  data/n/*.json   {len(nsz)} files, {mb(sum(nsz))} total, mean {sum(nsz) / max(1, len(nsz)) / 1e3:.1f} KB, "
          f"max {max(nsz or [0]) / 1e3:.1f} KB")
    print(f"  figures         {len(figs)} files")
    print(f"  total           {mb(total)}; largest file {biggest[1]} ({mb(biggest[0])})")


if __name__ == "__main__":
    main()
