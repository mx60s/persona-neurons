"""Persona-neuron explorer: browse a saved run (docs/run_format.md) and, optionally, probe the live model.

    python explorer/server.py --run runs/v1 --no-model                       # saved data only, no weights
    python explorer/server.py --run runs/v1 --model Qwen/Qwen3-8B --revision <sha> --adapters <dir>
    python explorer/server.py --run runs/fake --model ../work/tiny/base --adapters ../work/tiny/adapters

Without --model the model id and revision come from the run's manifest. --adapters is a directory holding o_amp,
o_sup, n_amp, n_sup, ctrl. Binds to 127.0.0.1 only. Writes only to runs/<id>/edits/.
"""
import argparse
import asyncio
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
        self._cache = {}
        self.write_lock = threading.Lock()
        os.makedirs(os.path.join(self.path, "edits", "cards"), exist_ok=True)

    def p(self, *parts):
        return os.path.join(self.path, *parts)

    def _json(self, rel, default=None):
        path = self.p(rel)
        if not os.path.exists(path):
            return default
        key = (rel, os.path.getmtime(path))
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            with open(path) as f:
                self._cache[key] = json.load(f)
        return self._cache[key]

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
        d = self.detail(nid)
        if not d:
            return {}
        key = ("ratios", nid, os.path.getmtime(self.p("neurons", "detail", fid(nid)[0] + ".json")))
        if key in self._cache:
            return self._cache[key]
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
        self._cache[key] = out
        return out

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
            "groups": list(GROUPS) + ["random_control"], "statuses": list(STATUSES)}


@app.get("/api/live/status")
def live_status():
    lv = LIVE["obj"]
    return {"loaded": lv is not None, "status": LIVE["status"], "error": LIVE["error"],
            "info": lv.info() if lv is not None else None}


@app.get("/api/neurons")
def neurons():
    edits = RUN.edit_statuses()
    rows = []
    for r in RUN.index():
        r = dict(r)
        if r["id"] in edits:
            r["card_status"], r["status_from"] = edits[r["id"]], "human edit"
        r["has_profile"] = os.path.exists(RUN.p("profiles", fid(r["id"])[0] + ".md"))
        r["example_ratio"] = RUN.example_ratios(r["id"], list((r.get("effects") or {}).keys()))
        rows.append(r)
    return rows


@app.get("/api/neuron")
def neuron(id: str):
    fid(id)
    row = next((r for r in neurons() if r["id"] == id), None)
    return {"id": id, "index": row, "detail": RUN.detail(id), "card": RUN.card(id), "profile": RUN.profile(id),
            "saved": RUN.saved(id, 50)}


@app.get("/api/plain")
def plain():
    """Plain-language layer; 404 when the run has no plain.json (the page then falls back to saved data)."""
    p = RUN.plain()
    if p is None:
        return JSONResponse({"error": "this run has no plain.json"}, status_code=404)
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
