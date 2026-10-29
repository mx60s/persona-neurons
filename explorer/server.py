"""Persona-neuron explorer: browse a saved run (docs/run_format.md) and, optionally, probe the live model.

    python explorer/server.py --run runs/v1 --no-model                       # saved data only, no weights
    python explorer/server.py --run runs/v1 --model Qwen/Qwen3-8B --revision <sha> --adapters <dir>
    python explorer/server.py --run runs/fake --model ../work/tiny/base --adapters ../work/tiny/adapters

Without --model the model id and revision come from the run's manifest. --adapters is a directory holding o_amp,
o_sup, n_amp, n_sup, ctrl. Binds to 127.0.0.1 only. Writes only to runs/<id>/edits/.
"""
import argparse
import asyncio
import collections
import datetime
import glob
import json
import os
import re
import sys
import threading
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from pn.states import parse_state, state_id  # noqa: E402

ID_RE = re.compile(r"^L(\d+)/N(\d+)$")
STATUSES = ("untested", "tested", "mixed", "rejected")
GROUPS = ("shared", "amp_primary", "sup_primary", "cross_trait", "control_assoc", "surprising")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fid(nid):
    m = ID_RE.match(nid or "")
    if not m:
        raise ValueError(f"bad neuron id {nid!r}")
    return f"L{m.group(1)}_N{m.group(2)}", int(m.group(1)), int(m.group(2))


class Run:
    """Read-only view of runs/<id>/, plus append-only writes under edits/."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.exists(os.path.join(self.path, "manifest.json")):
            raise SystemExit(f"{self.path} has no manifest.json")
        self._cache = collections.OrderedDict()   # rel -> (stamp, value), LRU
        self._small = {}                           # derived per-neuron values -> (stamp, value), never evicted
        self._finished = {"stamp": None, "offset": 0, "rows": [], "by_id": {}}
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
        os.makedirs(os.path.join(self.path, "edits", "cards"), exist_ok=True)

    def p(self, *parts):
        return os.path.join(self.path, *parts)

    def stamp(self, rel):
        try:
            st = os.stat(self.p(rel))
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _json(self, rel, default=None):
        """Parsed JSON, re-read when the file's mtime or size changes. A file caught mid-write keeps its last good
        value (or the default) instead of raising."""
        stamp = self.stamp(rel)
        if stamp is None:
            return default
        with self.read_lock:
            hit = self._cache.get(rel)
            if hit and hit[0] == stamp:
                self._cache.move_to_end(rel)
                return hit[1]
        try:
            with open(self.p(rel)) as f:
                val = json.load(f)
        except (ValueError, OSError):
            return hit[1] if hit else default
        with self.read_lock:
            self._cache[rel] = (stamp, val)
            self._cache.move_to_end(rel)
            while len(self._cache) > 400:
                self._cache.popitem(last=False)
        return val

    def manifest(self):
        return self._json("manifest.json")

    def index(self):
        return self._json("neurons/index.json", [])

    def detail(self, nid):
        f, _, _ = fid(nid)
        return self._json(f"neurons/detail/{f}.json")

    def profile(self, nid):
        f, _, _ = fid(nid)
        path = self.p("profiles", f"{f}.md")
        return open(path).read() if os.path.exists(path) else None

    def plain(self):
        """Plain-language layer (docs/plain_format.md), or None when the run has none."""
        return self._json("plain.json")

    def example_ratios(self, nid, states):
        """Fallback for neurons without a plain.json entry (same rule as pn/plain.py): median, over the ten saved
        top/quantile/top-diff example windows with the strongest base value at the focus token, of
        activation(state) / activation(base) at that token."""
        rel = f"neurons/detail/{fid(nid)[0]}.json"
        stamp = self.stamp(rel)
        if stamp is None:
            return {}
        key = ("ratios", nid, tuple(states))
        hit = self._small.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        d = self.detail(nid)
        if not d:
            return {}
        ex = [e for e in d.get("examples", []) if e.get("kind") in ("top_pos", "top_neg", "quantile", "top_diff")
              and e.get("focus") is not None and "base" in (e.get("acts") or {})]
        ex = sorted(ex, key=lambda e: -abs(e["acts"]["base"][e["focus"]]))[:10]
        out = {}
        for s in states:
            rs = sorted(e["acts"][s][e["focus"]] / e["acts"]["base"][e["focus"]] for e in ex
                        if s in e["acts"] and abs(e["acts"]["base"][e["focus"]]) > 1e-6)
            if rs:
                m = len(rs) // 2
                out[s] = rs[m] if len(rs) % 2 else (rs[m - 1] + rs[m]) / 2
        self._small[key] = (stamp, out)
        return out

    # ---- automated investigation (written by pn/investigate.py while it runs)

    def progress(self):
        return self._json("investigations/progress.json")

    def population_stats(self):
        return self._json("population_stats.json")

    def finished(self):
        """All rows of investigations/finished.jsonl in file order. Reads only the bytes appended since the last call;
        a partial last line is left for the next call. Rewritten or truncated files are re-read from the start."""
        rel = "investigations/finished.jsonl"
        stamp = self.stamp(rel)
        F = self._finished
        with self.read_lock:
            if stamp is None:
                F.update(stamp=None, offset=0, rows=[], by_id={})
                return F
            if stamp == F["stamp"]:
                return F
            if F["stamp"] is None or stamp[1] < F["offset"]:
                F.update(offset=0, rows=[], by_id={})
            with open(self.p(rel), "rb") as fh:
                fh.seek(F["offset"])
                chunk = fh.read()
            end = chunk.rfind(b"\n") + 1
            for line in chunk[:end].splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("id"):
                    F["rows"].append(r)
                    F["by_id"][r["id"]] = r
            F["offset"] += end
            F["stamp"] = stamp if end == len(chunk) else ("partial", F["offset"])
            return F

    def investigation(self, nid):
        f, _, _ = fid(nid)
        for name in (f, nid.replace("/", "_")):
            d = self._json(f"investigations/{name}.json")
            if d is not None:
                return d
        return None

    def posctrl_finished(self):
        """finished.jsonl of the positive-control run (the hand-profiled neurons through the same investigator), from
        --posctrl or the sibling directory <run>_posctrl; [] when absent."""
        d = (ARGS.posctrl if ARGS and getattr(ARGS, "posctrl", None) else self.path.rstrip("/") + "_posctrl")
        path = os.path.join(d, "investigations", "finished.jsonl")
        try:
            st = os.stat(path)
        except OSError:
            return []
        key = ("posctrl", path)
        hit = self._small.get(key)
        stamp = (st.st_mtime_ns, st.st_size)
        if hit and hit[0] == stamp:
            return hit[1]
        rows = []
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("id"):
                    rows.append(r)
        self._small[key] = (stamp, rows)
        return rows

    def run_complete(self):
        p = self.progress()
        if not p or not p.get("n_total"):
            return False
        return (p.get("n_done") or 0) + (p.get("n_failed") or 0) >= p["n_total"] and not p.get("n_running")

    def catalog(self):
        """Compact per-neuron records for the list endpoint, rebuilt when index, plain, finished or edits change."""
        edits = self.edit_statuses()
        fin = self.finished()
        key = (self.stamp("neurons/index.json"), self.stamp("plain.json"), fin["stamp"], len(fin["rows"]),
               tuple(sorted(edits.items())))
        hit = self._small.get("catalog")
        if hit and hit[0] == key:
            return hit[1]
        cat = build_catalog(self.index(), (self.plain() or {}).get("neurons") or {}, fin["by_id"], edits,
                            {os.path.basename(p)[:-3] for p in glob.glob(self.p("profiles", "*.md"))})
        self._small["catalog"] = (key, cat)
        return cat

    def prompts(self):
        return self._json("behavior/prompts.json", [])

    def edit_statuses(self):
        out = {}
        for path in glob.glob(self.p("edits", "cards", "*.json")):
            try:
                d = self._json(os.path.relpath(path, self.path))
                if d and d.get("status"):
                    out[d["id"]] = d["status"]
            except Exception:
                pass
        return out

    def card(self, nid):
        """Merged card: run versions + human edits, latest first. The original file is never modified."""
        f, _, _ = fid(nid)
        orig = self._json(f"neurons/cards/{f}.json")
        edits = self._json(f"edits/cards/{f}.json")
        if orig is None and edits is None:
            return None
        versions = [dict(v, source="run") for v in (orig or {}).get("versions", [])]
        versions += [dict(v, source="edits") for v in (edits or {}).get("versions", [])]
        versions.sort(key=lambda v: (v.get("created") or "", v.get("v", 0)), reverse=True)
        status = (edits or {}).get("status") or (orig or {}).get("status") or "untested"
        return {"id": nid, "status": status, "status_from": "human edit" if (edits or {}).get("status") else "run",
                "run_status": (orig or {}).get("status"), "versions": versions,
                "tests": [dict(t, source="run") for t in (orig or {}).get("tests", [])]
                + [dict(t, source="edits") for t in (edits or {}).get("tests", [])],
                "disagreements": (orig or {}).get("disagreements", []) + (edits or {}).get("disagreements", [])}

    def add_card_edit(self, nid, body):
        f, _, _ = fid(nid)
        status = body.get("status")
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self.write_lock:
            path = self.p("edits", "cards", f"{f}.json")
            edits = json.load(open(path)) if os.path.exists(path) else {"id": nid, "status": None, "versions": [],
                                                                          "tests": [], "disagreements": []}
            merged = self.card(nid) or {"versions": []}
            v = 1 + max([x.get("v", 0) for x in merged["versions"]] or [0])
            listy = lambda x: [s.strip() for s in x if str(s).strip()] if isinstance(x, list) else \
                [s.strip() for s in str(x or "").replace(",", "\n").split("\n") if s.strip()]
            ver = {"v": v, "author": (body.get("author") or "human").strip()[:80], "created": now(),
                   "kind": "human_edit", "blind": False,
                   "label": str(body.get("label", "")), "trigger_hypothesis": str(body.get("trigger_hypothesis", "")),
                   "output_hypothesis": str(body.get("output_hypothesis", "")),
                   "alternatives": [s.strip() for s in body.get("alternatives", []) if s.strip()]
                   if isinstance(body.get("alternatives"), list) else [s.strip() for s in str(body.get("alternatives", "")).split("\n") if s.strip()],
                   "adapter_changes": str(body.get("adapter_changes", "")),
                   "supporting": listy(body.get("supporting", [])), "contradicting": listy(body.get("contradicting", [])),
                   "proposed_tests": [s.strip() for s in body.get("proposed_tests", []) if s.strip()]
                   if isinstance(body.get("proposed_tests"), list) else [s.strip() for s in str(body.get("proposed_tests", "")).split("\n") if s.strip()],
                   "confidence": body.get("confidence") if body.get("confidence") in ("low", "medium", "high") else "low",
                   "no_coherent_hypothesis": bool(body.get("no_coherent_hypothesis", False)),
                   "note": str(body.get("note", "")), "status_set": status}
            edits["versions"].append(ver)
            if status:
                edits["status"] = status
            if body.get("disagreement"):
                edits["disagreements"].append(str(body["disagreement"]))
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(edits, fh, indent=1)
            os.replace(tmp, path)
        return self.card(nid)

    def save_result(self, rec):
        if "state" not in rec or "intervention" not in rec:
            raise ValueError("saved results must carry 'state' and 'intervention'")
        rec = dict(rec, saved=now(), run_id=self.manifest().get("run_id"))
        with self.write_lock, open(self.p("edits", "saved.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def saved(self, neuron=None, limit=200):
        path = self.p("edits", "saved.jsonl")
        if not os.path.exists(path):
            return []
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if neuron:
            rows = [r for r in rows if r.get("neuron") == neuron]
        return rows[-limit:][::-1]


# ---------------------------------------------------------------- app

ARGS = None
RUN = None
LIVE = {"obj": None, "status": "not loaded", "error": None}
app = FastAPI()


@app.middleware("http")
async def local_only(request, call_next):
    """Only this machine's own page: other sites (or a DNS name rebound to 127.0.0.1) can't reach the API."""
    from urllib.parse import urlparse
    port = ARGS.port if ARGS else 8791
    hosts = {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}
    origin = request.headers.get("origin")
    if (request.headers.get("host") not in hosts
            or (origin and urlparse(origin).netloc not in hosts)
            or request.headers.get("sec-fetch-site") in ("cross-site", "same-site")):
        return PlainTextResponse("forbidden", status_code=403)
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.exception_handler(KeyError)
async def key_error(request, exc):
    return JSONResponse({"error": f"unknown key {exc}"}, status_code=400)


def live_or_503():
    lv = LIVE["obj"]
    if lv is None:
        msg = "live model not loaded"
        if LIVE["status"] == "loading":
            msg += " (still loading)"
        elif LIVE["error"]:
            msg += f" (load failed: {LIVE['error']})"
        elif ARGS and ARGS.no_model:
            msg += " (server started with --no-model)"
        return None, JSONResponse({"error": msg, "live": False}, status_code=503)
    return lv, None


def norm_state(s):
    return state_id(parse_state(s))


def state_label(s):
    for st in RUN.manifest().get("states", []):
        if st["id"] == s:
            return st.get("label") or s
    return s


@app.get("/api/meta")
def meta():
    m = RUN.manifest()
    return {"run": m, "run_path": RUN.path, "live": live_status(), "prompts": RUN.prompts(),
            "groups": list(GROUPS) + ["random_control"], "statuses": list(STATUSES),
            "has_progress": RUN.progress() is not None, "run_complete": RUN.run_complete()}


@app.get("/api/live/status")
def live_status():
    lv = LIVE["obj"]
    return {"loaded": lv is not None, "status": LIVE["status"], "error": LIVE["error"],
            "info": lv.info() if lv is not None else None}


def sample_of(r, fin):
    """Which investigation sample a neuron belongs to: {"kind": "trait"|"random", "trait": ...} or None. Taken from the
    finished record when there is one, else from whatever marker the index row carries."""
    if fin:
        return {"kind": fin.get("kind") or "trait", "trait": fin.get("trait")}
    m = r.get("investigation") or r.get("sample") or r.get("sample_kind")
    if isinstance(m, dict):
        return {"kind": m.get("kind") or "trait", "trait": m.get("trait") or r.get("trait")}
    if isinstance(m, str) and m:
        return {"kind": "random" if "random" in m else "trait", "trait": r.get("trait")}
    k = str(r.get("kind") or "")
    if "sample" in k or "investigat" in k:
        return {"kind": "random" if "random" in k else "trait", "trait": r.get("trait")}
    return None


INV_KEYS = ("kind", "trait", "nickname", "one_liner", "status", "passes", "n_final_tests", "rounds", "cost_usd",
            "finished_at")


def neuron_row(r, edits, fin_by_id):
    r = dict(r)
    if r["id"] in edits:
        r["card_status"], r["status_from"] = edits[r["id"]], "human edit"
    r["has_profile"] = os.path.exists(RUN.p("profiles", fid(r["id"])[0] + ".md"))
    r["example_ratio"] = RUN.example_ratios(r["id"], list((r.get("effects") or {}).keys()))
    fin = fin_by_id.get(r["id"])
    r["inv"] = {k: fin.get(k) for k in INV_KEYS} if fin else None
    r["sample"] = sample_of(r, fin)
    return r


def stub_row(fin):
    """Neuron that finished an investigation but has no row in index.json (yet)."""
    _, layer, n = fid(fin["id"])
    return {"id": fin["id"], "layer": layer, "n": n, "kind": "sample", "effects": {}}


# ---------------------------------------------------------------- neuron catalog (list endpoint)

POLES = {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}
VERDICTS = ("held_up", "partly", "not_held", "no_coherent", "failed", "none")
SAMPLES = ("profiled", "trait", "neuroticism", "openness", "both", "random", "v1")
CHANGES = ("up", "down", "reshaped", "offset")
WHERES = ("reply_start", "text")
OPPS = ("opposite", "same")
DIMS = ("sample", "verdict", "change", "where", "opp")


def _change_in(r, poles):
    ct = r.get("change_type") or {}
    eff = r.get("effects") or {}
    cands = [p for p in poles if p in ct]
    if not cands:
        return None
    s = max(cands, key=lambda p: (eff.get(p) or {}).get("z_rms") or 0)
    t = ct[s]
    typ = t.get("type")
    if typ == "gain":
        typ = "up" if (t.get("gain") or 0) > 1 else "down"
    return {"type": typ, "gain": t.get("gain"), "corr": t.get("corr_with_base"), "pole": s}


def build_catalog(index, plain, fin_by_id, edits, profile_files):
    polar = [s for t in POLES.values() for s in t]
    cat = []
    for r in index:
        nid = r.get("id")
        try:
            f, _, _ = fid(nid)
        except ValueError:
            continue
        P = plain.get(nid) or {}
        fin = fin_by_id.get(nid)
        ct = r.get("change_type")
        pop = bool(ct) or bool(r.get("in_population")) or fin is not None
        rand = r.get("kind") == "random_control"
        traits = set()
        if pop and not rand:
            traits = {t for t in (r.get("selected_for") or []) if t in POLES}
            if fin and fin.get("kind") != "random" and fin.get("trait") in POLES:
                traits.add(fin["trait"])
        samples = set()
        if r.get("detailed") or f in profile_files:
            samples.add("profiled")
        if traits:
            samples.add("trait")
            samples.update(traits)
            if len(traits) > 1:
                samples.add("both")
        if rand and pop:
            samples.add("random")
        if not ct:
            samples.add("v1")
        # kind of change under the pole that changes it most, per context (a trait, or "any")
        change = {}
        for t in traits:
            change[t] = _change_in(r, POLES[t])
        any_poles = [p for t in sorted(traits) for p in POLES[t]] or polar
        change["any"] = _change_in(r, any_poles)
        cas = r.get("corr_amp_sup") or {}
        opp = {t: ("opposite" if cas[t] < 0 else "same") for t in traits if cas.get(t) is not None}
        main_t = next((t for t, ps in POLES.items() if change["any"] and change["any"]["pole"] in ps), None)
        if main_t is None and traits:
            eff0 = r.get("effects") or {}
            main_t = max(sorted(traits), key=lambda t: max((eff0.get(p) or {}).get("z_rms") or 0 for p in POLES[t]))
        opp["any"] = opp.get(main_t) if main_t in traits else None
        td = r.get("template_driven")
        where = None if td is None else ("reply_start" if td else "text")
        reviewed = P.get("source") != "automated" and bool(P.get("nickname") or P.get("one_liner"))
        hand, hand_src = None, None
        if nid in edits:
            hand, hand_src = edits[nid], "human edit"
        elif reviewed:
            hand, hand_src = (P.get("checked") or {}).get("status") or "untested", "hand"
        if reviewed and P.get("nickname"):
            name, name_src = P["nickname"], "hand"
        elif fin and fin.get("nickname"):
            name, name_src = fin["nickname"], "investigator"
        elif P.get("nickname"):
            name, name_src = P["nickname"], "auto"
        else:
            name, name_src = nid, "id"
        if reviewed and P.get("one_liner"):
            one, one_src = P["one_liner"], "hand"
        elif fin and fin.get("one_liner"):
            one, one_src = fin["one_liner"], "investigator"
        elif P.get("one_liner"):
            one, one_src = P["one_liner"], "auto"
        else:
            one, one_src = r.get("label") or "", "auto"
        verdict = (fin.get("status") or "no_coherent") if fin else ("failed" if pop else "none")
        eff = r.get("effects") or {}
        effect = max([(eff.get(s) or {}).get("z_rms") or 0 for s in polar] or [0])
        arrows = [[e.get("state"), e.get("direction"), e.get("ratio")] for e in (P.get("adapter_effects") or [])]
        search = " ".join(str(x or "") for x in (nid, name, one, P.get("nickname"), P.get("one_liner"),
                                                  (fin or {}).get("nickname"), (fin or {}).get("one_liner"))).lower()
        cat.append({
            "id": nid, "layer": r.get("layer"), "n": r.get("n"), "kind": r.get("kind"), "samples": samples,
            "traits": sorted(traits), "name": name, "name_src": name_src, "one": one, "one_src": one_src,
            "hand": hand, "hand_src": hand_src, "verdict": verdict,
            "inv": {k: fin.get(k) for k in ("status", "passes", "n_final_tests", "rounds")} if fin else None,
            "change": change, "opp": opp, "where": where, "effect": effect, "arrows": arrows,
            "search": search,
            "tech": {"z_mean": {s: (eff.get(s) or {}).get("z_mean") for s in eff if not re.search(r"[+-]0\.5$|-1$", s)},
                     "card_status": edits.get(nid) or r.get("card_status"), "detailed": bool(r.get("detailed")),
                     "template_driven": td,
                     "outlier": any(isinstance(v, dict) and (v.get("top_share") or 0) > 0.6 for v in (r.get("robust") or {}).values())},
        })
    return cat


def _csv(x, allowed):
    vals = [v for v in (x or "").split(",") if v]
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"unknown value(s) {bad}; allowed: {list(allowed)}")
    return set(vals)


def _ctx(sample):
    return sample if sample in POLES else "any"


def _dim_values(c, ctx):
    """The value(s) of each facet dimension for one catalog record."""
    ch = c["change"].get(ctx) if ctx in c["change"] else c["change"].get("any")
    op = c["opp"].get(ctx) if ctx in c["opp"] else c["opp"].get("any")
    return {"sample": c["samples"], "verdict": {c["verdict"]}, "change": {ch["type"]} if ch else set(),
            "where": {c["where"]} if c["where"] else set(), "opp": {op} if op else set()}


VERDICT_RANK = {"held_up": 0, "partly": 1, "not_held": 2, "no_coherent": 3, "failed": 4, "none": 5}


def _sort_key(sort, ctx):
    if sort == "default":
        return lambda c: ((0 if "profiled" in c["samples"] else 1 + min(VERDICT_RANK[c["verdict"]], 2)), -c["effect"])
    if sort == "verdict":
        return lambda c: (VERDICT_RANK[c["verdict"]], -((c["inv"] or {}).get("passes") or 0), -c["effect"])
    if sort == "effect":
        return lambda c: -c["effect"]
    if sort == "layer":
        return lambda c: (c["layer"] or 0, c["n"] or 0)
    if sort == "passes":
        return lambda c: (-((c["inv"] or {}).get("passes") if c["inv"] else -1),
                          (c["inv"] or {}).get("n_final_tests") or 0, -c["effect"])
    if sort.startswith("z:"):
        s = sort[2:]
        return lambda c: -(c["tech"]["z_mean"].get(s) if c["tech"]["z_mean"].get(s) is not None else -1e9)
    raise ValueError(f"unknown sort {sort!r}")


def list_rows(cat, sample="", verdict="", change="", where="", opp="", lmin=None, lmax=None, q="", sort="default",
              dir="", page=1, size=50):
    if sample and sample not in SAMPLES:
        raise ValueError(f"unknown sample {sample!r}; allowed: {list(SAMPLES)}")
    want = {"sample": {sample} if sample else set(), "verdict": _csv(verdict, VERDICTS),
            "change": _csv(change, CHANGES), "where": _csv(where, WHERES), "opp": _csv(opp, OPPS)}
    ctx = _ctx(sample)
    terms = [t for t in (q or "").lower().split() if t]
    facets = {d: collections.Counter() for d in DIMS}
    hits = []
    for c in cat:
        if lmin is not None and (c["layer"] or 0) < lmin:
            continue
        if lmax is not None and (c["layer"] or 0) > lmax:
            continue
        if terms and not all(t in c["search"] for t in terms):
            continue
        vals = _dim_values(c, ctx)
        ok = {d: (not want[d]) or bool(vals[d] & want[d]) for d in DIMS}
        nfail = sum(not v for v in ok.values())
        if nfail == 0:
            hits.append(c)
        for d in DIMS:
            if nfail == 0 or (nfail == 1 and not ok[d]):
                facets[d].update(vals[d])
                facets[d]["_all"] += 1
    hits.sort(key=_sort_key(sort, ctx), reverse=(dir == "asc"))
    size = min(max(int(size), 1), 200)
    pages = max(1, -(-len(hits) // size))
    page = min(max(int(page), 1), pages)
    out = []
    for c in hits[(page - 1) * size: page * size]:
        ch = c["change"].get(ctx) if ctx in c["change"] else c["change"].get("any")
        op = c["opp"].get(ctx) if ctx in c["opp"] else c["opp"].get("any")
        out.append({k: c[k] for k in ("id", "layer", "kind", "traits", "name", "name_src", "one", "one_src", "hand",
                                      "hand_src", "verdict", "inv", "where", "effect", "arrows", "tech")}
                   | {"samples": sorted(c["samples"]), "change": ch, "opp": op})
    return {"total": len(hits), "page": page, "pages": pages, "size": size, "rows": out, "context": ctx,
            "facets": {d: dict(v) for d, v in facets.items()}, "n_all": len(cat),
            "layers": [min((c["layer"] for c in cat if c["layer"] is not None), default=0),
                       max((c["layer"] for c in cat if c["layer"] is not None), default=0)]}


@app.get("/api/list")
def list_endpoint(sample: str = "", verdict: str = "", change: str = "", where: str = "", opp: str = "",
                  lmin: int = None, lmax: int = None, q: str = "", sort: str = "default", dir: str = "",
                  page: int = 1, size: int = 50):
    """Filtered, sorted, paginated neuron list with facet counts (each dimension counted with the other filters
    applied). Comma-separated values within a dimension are OR'ed."""
    return list_rows(RUN.catalog(), sample, verdict, change, where, opp, lmin, lmax, q, sort, dir, page, size)


def _verdict_counts(rows_or_cats, key):
    c = collections.Counter(key(x) for x in rows_or_cats)
    n = sum(c[v] for v in ("held_up", "partly", "not_held", "no_coherent"))
    return {"n": n, "held_up": c["held_up"], "partly": c["partly"], "not_held": c["not_held"] + c["no_coherent"],
            "failed": c["failed"]}


POP_KEYS = (("n", lambda k: k == "n"), ("rescaled", lambda k: "(gain)" in k or k.startswith("same response")),
            ("up", lambda k: "which turned up" in k), ("down", lambda k: "which turned down" in k),
            ("reshaped", lambda k: "reshaped" in k), ("offset", lambda k: "offset" in k),
            ("reply_start", lambda k: "reply start" in k), ("opposite", lambda k: "opposite" in k))


@app.get("/api/results")
def results():
    """Numbers for the Start page result tables: measured properties (population_stats.json, or computed from the
    index when it is missing) and investigation verdicts (finished.jsonl joined with the index, so the counts match
    the list filters), plus the positive-control run when present."""
    cat = RUN.catalog()
    st = RUN.population_stats()
    pop = []
    for name in ("neuroticism", "openness", "random"):
        g = (st or {}).get(name)
        if isinstance(g, dict):
            row = {"group": name, "source": "population_stats.json"}
            for out, match in POP_KEYS:
                k = next((k for k in g if match(k.strip())), None)
                row[out] = g.get(k) if k else None
        else:
            cs = [c for c in cat if name in c["samples"]]
            if not cs:
                continue
            ctx = _ctx(name)
            types = collections.Counter((_dim_values(c, ctx)["change"] or {None}).pop() for c in cs)
            row = {"group": name, "source": "index", "n": len(cs), "up": types["up"], "down": types["down"],
                   "rescaled": types["up"] + types["down"], "reshaped": types["reshaped"], "offset": types["offset"],
                   "reply_start": sum(c["where"] == "reply_start" for c in cs),
                   "opposite": sum(_dim_values(c, ctx)["opp"] == {"opposite"} for c in cs) if name in POLES else None}
        pop.append(row)
    inv = []
    for name in ("trait", "neuroticism", "openness", "random"):
        cs = [c for c in cat if name in c["samples"]]
        if cs and any(c["verdict"] != "none" for c in cs):
            inv.append({"group": name, **_verdict_counts(cs, lambda c: c["verdict"])})
    pc = RUN.posctrl_finished()
    if pc:
        inv.append({"group": "posctrl", **_verdict_counts(pc, lambda r: r.get("status") or "no_coherent")})
    p = RUN.progress()
    return {"population": pop, "population_notes": (st or {}).get("notes"), "investigation": inv,
            "progress": {k: p.get(k) for k in ("n_total", "n_done", "n_failed", "n_running", "cost_usd", "started",
                                               "updated")} if p else None,
            "run_complete": RUN.run_complete(), "n_neurons": len(cat),
            "n_profiled": sum("profiled" in c["samples"] for c in cat),
            "samples": dict(collections.Counter(s for c in cat for s in c["samples"]))}


@app.get("/api/neurons")
def neurons():
    edits, fin = RUN.edit_statuses(), RUN.finished()["by_id"]
    rows, seen = [], set()
    for r in RUN.index():
        try:
            rows.append(neuron_row(r, edits, fin))
            seen.add(r["id"])
        except (ValueError, KeyError):
            continue
    for nid, f in list(fin.items()):
        if nid not in seen:
            try:
                rows.append(neuron_row(stub_row(f), edits, fin))
            except ValueError:
                pass
    return rows


@app.get("/api/neuron")
def neuron(id: str):
    fid(id)
    edits, fin = RUN.edit_statuses(), RUN.finished()["by_id"]
    raw = next((r for r in RUN.index() if r.get("id") == id), None)
    if raw is None and id in fin:
        raw = stub_row(fin[id])
    row = neuron_row(raw, edits, fin) if raw else None
    plain_entry = ((RUN.plain() or {}).get("neurons") or {}).get(id)
    posctrl = next((r for r in RUN.posctrl_finished() if r.get("id") == id), None)
    return {"id": id, "index": row, "detail": RUN.detail(id), "card": RUN.card(id), "profile": RUN.profile(id),
            "saved": RUN.saved(id, 50), "investigation": RUN.investigation(id), "plain": plain_entry,
            "posctrl": posctrl}


@app.get("/api/progress")
def progress():
    """investigations/progress.json as written by the investigator, plus the server's clock for staleness."""
    p = RUN.progress()
    if p is None:
        return JSONResponse({"error": "no investigation run in progress (investigations/progress.json missing)"},
                            status_code=404)
    return dict(p, server_now=now(), n_finished_lines=len(RUN.finished()["rows"]))


@app.get("/api/finished")
def finished(limit: int = 30, offset: int = 0):
    """Finished neurons, newest first: rows [offset, offset + limit) counted from the end of finished.jsonl."""
    rows = RUN.finished()["rows"]
    limit, offset = min(max(limit, 1), 1000), max(offset, 0)
    end = max(len(rows) - offset, 0)
    return {"total": len(rows), "offset": offset, "rows": rows[max(end - limit, 0):end][::-1]}


@app.get("/api/investigation")
def investigation(id: str):
    d = RUN.investigation(id)
    if d is None:
        return JSONResponse({"error": f"no investigation for {id}"}, status_code=404)
    return d


@app.get("/api/population_stats")
def population_stats():
    d = RUN.population_stats()
    if d is None:
        return JSONResponse({"error": "no population_stats.json in this run"}, status_code=404)
    return d


@app.get("/api/plain")
def plain(lite: int = 0):
    """Plain-language layer; 404 when the run has no plain.json (the page then falls back to saved data). lite=1 keeps
    only the hand-written neuron entries (the browser fetches the others with each neuron)."""
    p = RUN.plain()
    if p is None:
        return JSONResponse({"error": "this run has no plain.json"}, status_code=404)
    if lite:
        keep = {k: v for k, v in (p.get("neurons") or {}).items() if v.get("source") != "automated" or v.get("featured")}
        return dict(p, neurons=keep)
    return p


@app.get("/api/profile")
def profile(id: str):
    text = RUN.profile(id)
    if text is None:
        return JSONResponse({"error": f"no profile for {id}"}, status_code=404)
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


FIG_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(png|jpg|jpeg|svg|gif)$")


@app.get("/figures/{name}")
def figure(name: str):
    """Figures from runs/<id>/figures/ if present, else the repository's figures/."""
    if not FIG_RE.match(name):
        return JSONResponse({"error": "bad figure name"}, status_code=400)
    for d in (RUN.p("figures"), os.path.join(os.path.dirname(HERE), "figures")):
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return FileResponse(path)
    return JSONResponse({"error": f"no figure {name}"}, status_code=404)


@app.get("/api/card")
def card(id: str):
    return RUN.card(id) or {"id": id, "status": "untested", "versions": [], "tests": [], "disagreements": []}


@app.post("/api/card")
async def card_edit(req: Request):
    b = await req.json()
    return RUN.add_card_edit(b["id"], b)


@app.post("/api/save")
async def save(req: Request):
    b = await req.json()
    return RUN.save_result(b)


@app.get("/api/saved")
def saved(neuron: str = None):
    return RUN.saved(neuron)


# ---------------------------------------------------------------- live endpoints

def iv_dict(nid, mode, value):
    if mode == "zero":
        return {"neuron": nid, "mode": "zero", "value": 0.0, "unit": "x"}
    if mode == "mul":
        return {"neuron": nid, "mode": "mul", "value": float(value), "unit": "x"}
    return {"neuron": nid, "mode": "add", "value": float(value), "unit": "resid_p50"}


def variant_ivs(nid, mode, value):
    """Interventions for the comparison columns besides 'unmodified': add/mul push both ways, zero has one."""
    v = abs(float(value))
    if mode == "zero":
        return [iv_dict(nid, "zero", 0)]
    if mode == "mul":
        return [iv_dict(nid, "mul", float(value)), iv_dict(nid, "mul", -float(value))]
    if mode == "add":
        return [iv_dict(nid, "add", v), iv_dict(nid, "add", -v)]
    raise ValueError(f"mode must be add, mul or zero, got {mode!r}")


@app.get("/api/live/calibration")
async def live_calibration(neuron: str, state: str = "base"):
    lv, err = live_or_503()
    if err:
        return err
    _, layer, n = fid(neuron)
    return await asyncio.to_thread(lv.calibration, layer, n, state)


@app.post("/api/live/read")
async def live_read(req: Request):
    lv, err = live_or_503()
    if err:
        return err
    b = await req.json()
    _, layer, n = fid(b["neuron"])
    states = b.get("states") or ["base"]
    res = await asyncio.to_thread(lv.read, layer, n, b["text"], states, bool(b.get("chat", False)))
    res.update(neuron=b["neuron"], intervention=None, labels={s: state_label(s) for s in res["states"]})
    return res


@app.post("/api/live/next")
async def live_next(req: Request):
    lv, err = live_or_503()
    if err:
        return err
    b = await req.json()
    nid = b["neuron"]
    _, layer, n = fid(nid)
    ivs = variant_ivs(nid, b.get("mode", "add"), b.get("value", 1.0))
    state = norm_state(b.get("state", "base"))
    res = await asyncio.to_thread(lv.next_token, state, layer, n, b["text"], ivs, bool(b.get("chat", True)),
                                  b.get("prefix", ""))
    res["calibration"] = await asyncio.to_thread(lv.calibration, layer, n, state)
    res.update(neuron=nid, state_label=state_label(state), text=b["text"], prefix=b.get("prefix", ""))
    return res


@app.get("/api/live/lens")
async def live_lens(neuron: str, state: str = "base"):
    lv, err = live_or_503()
    if err:
        return err
    _, layer, n = fid(neuron)
    return await asyncio.to_thread(lv.lens, layer, n, state)


@app.get("/api/live/connections")
async def live_connections(neuron: str, state: str = "base", k: int = 16):
    lv, err = live_or_503()
    if err:
        return err
    _, layer, n = fid(neuron)
    return await asyncio.to_thread(lv.connections, layer, n, state, min(max(k, 1), 64))


def sse(req, work):
    """Runs work(emit, cancel) in a thread and streams each emitted dict as a server-sent event."""
    loop, q, cancel = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def run():
        try:
            work(lambda ev: loop.call_soon_threadsafe(q.put_nowait, ev), cancel)
        except Exception as e:
            traceback.print_exc()
            loop.call_soon_threadsafe(q.put_nowait, {"error": str(e)})
        loop.call_soon_threadsafe(q.put_nowait, {"end": True})

    threading.Thread(target=run, daemon=True).start()

    async def events():
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("end"):
                    break
                if await req.is_disconnected():
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/live/chat")
async def live_chat(req: Request):
    """Matched chats: the same user message in each model state, greedy, no neuron intervention."""
    lv, err = live_or_503()
    if err:
        return err
    b = await req.json()
    states = list(dict.fromkeys(lv.check_state(s) for s in (b.get("states") or ["base"])))[:16]
    msgs = [{"role": "user", "content": b["message"]}]
    max_new = int(b.get("max_new", 128))

    def work(emit, cancel):
        emit({"columns": [{"state": s, "label": state_label(s), "intervention": None} for s in states]})
        for i, s in enumerate(states):
            for row, delta in lv.generate(s, [msgs], [None], max_new=max_new, cancel=cancel):
                if row is None:
                    break
                emit({"col": i, "t": delta})
            emit({"col": i, "done": True})
            if cancel.is_set():
                break

    return sse(req, work)


@app.post("/api/live/steer")
async def live_steer(req: Request):
    """Neuron intervention within one model state: columns +, unmodified, - (or zeroed, unmodified)."""
    lv, err = live_or_503()
    if err:
        return err
    b = await req.json()
    nid = b["neuron"]
    _, layer, n = fid(nid)
    state = lv.check_state(b.get("state", "base"))
    mode = b.get("mode", "add")
    ivs = variant_ivs(nid, mode, b.get("value", 1.0))
    cols = [ivs[0], None] + ivs[1:]
    msgs = [{"role": "user", "content": b["message"]}]
    max_new = int(b.get("max_new", 128))

    def work(emit, cancel):
        cal = lv.calibration(layer, n, state)
        emit({"state": state, "state_label": state_label(state), "calibration": cal,
              "columns": [{"state": state, "intervention": c} for c in cols]})
        for row, delta in lv.generate(state, [msgs] * len(cols), cols, layer, n, max_new=max_new, cancel=cancel):
            if row is None:
                break
            emit({"col": row, "t": delta})

    return sse(req, work)


@app.get("/")
def root():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


def load_live():
    LIVE["status"] = "loading"
    try:
        from live import Live
        m = RUN.manifest().get("model", {})
        model = ARGS.model or m.get("id")
        revision = ARGS.revision if ARGS.model else (ARGS.revision or m.get("revision"))
        LIVE["obj"] = Live(model, ARGS.adapters, revision=revision, device=ARGS.device, dtype=ARGS.dtype,
                           tokenizer_id=ARGS.tokenizer)
        LIVE["status"] = "ready"
        print(f"live model ready: {model} @ {revision}, adapters {list(LIVE['obj'].adapters)}", flush=True)
    except Exception as e:
        traceback.print_exc()
        LIVE["status"], LIVE["error"] = "failed", f"{type(e).__name__}: {e}"


def main():
    global ARGS, RUN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<run_id>")
    ap.add_argument("--model", default=None, help="base model id or path (default: manifest model.id)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapters", default=None, help="dir with o_amp, o_sup, n_amp, n_sup, ctrl adapter dirs")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--no-model", action="store_true", help="browse saved data only; never load weights")
    ap.add_argument("--posctrl", default=None, help="positive-control run dir (default: <run>_posctrl if present)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8791)))
    ARGS = ap.parse_args()
    RUN = Run(ARGS.run)
    if ARGS.no_model:
        LIVE["status"] = "disabled (--no-model)"
    else:
        threading.Thread(target=load_live, daemon=True).start()
    print(f"run {RUN.path}\nhttp://127.0.0.1:{ARGS.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
