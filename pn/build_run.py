"""Assembles a browsable run (docs/run_format.md) from the corpus, traces, metrics and the candidate list.

    python -m pn.build_run --run runs/v1 --corpus data/corpus.jsonl --traces /workspace/out/traces_v1 \
        --candidates /workspace/out/pick_v1/candidates.json [--model ... --adapters ...]

Loads the model once for the logit lens per state and the residual-unit calibration. Example windows are split into
discovery and validation by document (a fixed hash), so held-out windows never share a document with discovery ones.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from pn.collect import ADAPTER_DIRS, DOMAINS
from pn.intervene import Calibration
from pn.states import StateModel

PRE, POST = 28, 8  # tokens of context before and after the focus token
TRAIT_ADAPTERS = {"openness": ("o_amp", "o_sup"), "neuroticism": ("n_amp", "n_sup")}
STATE_LABELS = {"base": "base", "ctrl+1": "neutral control x1"}
for a, lab in (("o_amp", "openness amplifier"), ("o_sup", "openness suppressor"), ("n_amp", "neuroticism amplifier"),
               ("n_sup", "neuroticism suppressor")):
    for c in ("-1", "-0.5", "+0.5", "+1"):
        STATE_LABELS[f"{a}{c}"] = f"{lab} x{c.lstrip('+')}"


def split_of(doc_id):
    return "validation" if int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 10 < 3 else "discovery"


class Builder:
    def __init__(self, args):
        self.args = args
        self.docs = [json.loads(l) for l in open(args.corpus)]
        tm = json.load(open(os.path.join(args.traces, "meta.json")))
        self.tmeta = tm
        self.states = tm["states"]
        self.neuron_k = {f"L{l}/N{n}": k for k, (l, n) in enumerate(tm["neurons"])}
        # [K, S, T] in RAM, one contiguous block per neuron (column reads from a [S, T, K] file are slow at scale)
        load = lambda name: np.ascontiguousarray(np.transpose(np.load(os.path.join(args.traces, name), mmap_mode="r"), (2, 0, 1)))  # noqa: E731
        self.act, self.gate, self.up = load("act.npy"), load("gate.npy"), load("up.npy")
        self.offsets = np.array(tm["offsets"])
        self.lengths = np.array(tm["lengths"])
        T = int(self.offsets[-1] + self.lengths[-1])
        self.doc_of = np.zeros(T, np.int32)
        self.pos_of = np.zeros(T, np.int32)
        for i, (o, n) in enumerate(zip(self.offsets, self.lengths)):
            self.doc_of[o:o + n] = i
            self.pos_of[o:o + n] = np.arange(n)
        self.valid = self.pos_of > 0
        self.seg = np.full(T, 2, np.int8)
        self.dom = np.zeros(T, np.int8)
        for i, d in enumerate(self.docs):
            o, n = self.offsets[i], self.lengths[i]
            self.dom[o:o + n] = DOMAINS.index(d["domain"])
            if d.get("boundary") is not None:
                self.seg[o:o + n] = (np.arange(n) >= d["boundary"]).astype(np.int8)
        self.split = np.array([split_of(d["doc_id"]) for d in self.docs])
        adapters = {k: os.path.join(args.adapters, k if os.path.isdir(os.path.join(args.adapters, k)) else v)
                    for k, v in ADAPTER_DIRS.items()}
        self.sm = StateModel(args.model, adapters, revision=args.revision, device=args.device)
        self.tok = self.sm.tok
        self.calib = Calibration(self.sm)
        self.calib.measure()
        self.piece = {}
        self.rng = np.random.default_rng(0)

    def token_str(self, t):
        if t not in self.piece:
            self.piece[t] = self.tok.decode([t])
        return self.piece[t]

    def window(self, k, flat, kind, show_states):
        i = int(self.doc_of[flat])
        d = self.docs[i]
        p = int(self.pos_of[flat])
        o = int(self.offsets[i])
        a, b = max(1, p - PRE), min(len(d["ids"]), p + POST + 1)
        ids = d["ids"][a:b]
        bnd = d.get("boundary")
        ex = {"doc_id": d["doc_id"], "domain": d["domain"], "source": d["source"], "kind": kind,
              "split": str(self.split[i]), "tokens": [self.token_str(t) for t in ids], "start": a, "focus": p - a,
              "boundary": (bnd - a) if bnd is not None and a < bnd < b else None,
              "boundary_before_window": bnd is not None and bnd <= a,
              "acts": {st: np.round(self.act[k, self.states.index(st), o + a:o + b].astype(np.float32), 4).tolist()
                       for st in show_states}}
        if self.args.gate_up:
            ex["gate"] = {st: np.round(self.gate[k, self.states.index(st), o + a:o + b].astype(np.float32), 4).tolist()
                          for st in show_states[:5]}
            ex["up"] = {st: np.round(self.up[k, self.states.index(st), o + a:o + b].astype(np.float32), 4).tolist()
                        for st in show_states[:5]}
        return ex

    def pick_positions(self, score, n, used_docs, max_per_doc=2, domains_spread=True):
        order = np.argsort(-score)
        out, per_doc, per_dom = [], {}, {}
        for f in order:
            if not np.isfinite(score[f]) or score[f] == -np.inf:
                break
            i = int(self.doc_of[f])
            if per_doc.get(i, 0) >= max_per_doc:
                continue
            if any(abs(int(f) - int(g)) <= PRE for g in out):
                continue
            dm = int(self.dom[f])
            if domains_spread and per_dom.get(dm, 0) >= max(2, n // 3):
                continue
            out.append(int(f))
            per_doc[i] = per_doc.get(i, 0) + 1
            per_dom[dm] = per_dom.get(dm, 0) + 1
            if len(out) >= n:
                break
        return out

    def neuron(self, rec):
        k = self.neuron_k[rec["id"]]
        L, n = rec["layer"], rec["n"]
        S = self.states
        A = {st: self.act[k, S.index(st)].astype(np.float32) for st in S}
        base = A["base"]
        v = self.valid
        traits = [t for t in TRAIT_ADAPTERS if t in rec["selected_for"]] or list(TRAIT_ADAPTERS)
        poles = [s for t in TRAIT_ADAPTERS for s in (f"{TRAIT_ADAPTERS[t][0]}+1", f"{TRAIT_ADAPTERS[t][1]}+1")]
        show = ["base"] + poles + ["ctrl+1"]
        show = [s for s in show if s in S]
        examples = []

        def add(positions, kind):
            for f in positions:
                examples.append(self.window(k, f, kind, show))

        neg_inf = np.where(v, 0, -np.inf)
        add(self.pick_positions(base + neg_inf, 8, None), "top_pos")
        add(self.pick_positions(-base + neg_inf, 6, None), "top_neg")
        for t in traits:
            for ad in TRAIT_ADAPTERS[t]:
                st = f"{ad}+1"
                d = A[st] - base
                add(self.pick_positions(np.abs(d) + neg_inf, 4, None), "top_diff")
        dc = A["ctrl+1"] - base
        add(self.pick_positions(np.abs(dc) + neg_inf, 2, None), "top_diff")
        vb = base[v]
        for q in (0.25, 0.5, 0.75, 0.9, 0.97, 0.995):
            target = np.quantile(vb, q)
            near = np.where(v, -np.abs(base - target), -np.inf)
            cand = np.argsort(-near)[:200]
            f = int(self.rng.choice(cand))
            add([f], "quantile")
        add([int(x) for x in self.rng.choice(np.flatnonzero(v), 5, replace=False)], "random")
        # dedupe identical windows
        seen, uniq = set(), []
        for e in examples:
            key = (e["doc_id"], e["start"])
            if key in seen:
                continue
            seen.add(key)
            e["ex_id"] = f"{rec['id'].replace('/', '_')}:e{len(uniq):02d}"
            uniq.append(e)

        # trajectories with document-bootstrap CIs
        di = self.doc_of[v]
        doc_n = np.bincount(di, minlength=len(self.docs)).astype(np.float64)
        B = 200
        counts = np.stack([np.bincount(self.rng.integers(0, len(self.docs), len(self.docs)), minlength=len(self.docs))
                           for _ in range(B)]).astype(np.float64)
        W = counts * doc_n[None]
        denom = rec["denominator"]

        def stat(st):
            x = A[st][v]
            dsum = np.bincount(di, weights=x - base[v], minlength=len(self.docs))
            dmean = dsum / np.maximum(doc_n, 1)
            bs = (W @ dmean) / W.sum(1)
            md = float((x - base[v]).mean())
            return {"state": st, "label": STATE_LABELS.get(st, st), "mean": float(x.mean()),
                    "rms": float(np.sqrt((x ** 2).mean())), "mean_diff": md,
                    "rms_diff": float(np.sqrt(((x - base[v]) ** 2).mean())), "z_mean": md / denom,
                    "z_rms": float(np.sqrt(((x - base[v]) ** 2).mean())) / denom,
                    "ci90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]}
        traj = {}
        for t, (amp, sup) in TRAIT_ADAPTERS.items():
            seq = [f"{amp}-1", f"{amp}-0.5", "base", f"{amp}+0.5", f"{amp}+1", f"{sup}+1", "ctrl+1"]
            traj[t] = [stat(s) for s in seq if s in S]
        by_domain, by_segment = {}, {}
        for st in S:
            if st == "base":
                continue
            d = (A[st] - base)
            by_domain[st] = {dm: {"mean_diff": float(d[v & (self.dom == j)].mean()) if (v & (self.dom == j)).any() else None,
                                  "n_tok": int((v & (self.dom == j)).sum())} for j, dm in enumerate(DOMAINS)}
            by_segment[st] = {sg: float(d[v & (self.seg == j)].mean()) if (v & (self.seg == j)).any() else None
                              for j, sg in enumerate(("prompt", "continuation", "nosplit"))}
        qs = {f"p{str(q).replace('0.', '')}": float(np.quantile(vb, q))
              for q in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)}
        lens = {}
        with torch.inference_mode():
            for st in show:
                dcol = self.sm.down_col(L, n, state=st).to(self.sm.dtype).to(self.sm.device)
                logits = self.sm.model.lm_head(self.sm.lm.norm(dcol[None, None]))[0, 0].float()
                lens[st] = {"up": [self.token_str(t) for t in logits.topk(12).indices.tolist()],
                            "down": [self.token_str(t) for t in (-logits).topk(12).indices.tolist()]}
        d_base = self.sm.down_col(L, n, state="base")
        weights = {"down_norm": {}, "down_cos_to_base": {}, "gate_cos_to_base": {}, "up_cos_to_base": {}}
        g0, u0 = self.sm.in_rows(L, n, state="base")
        for st in show:
            dc_ = self.sm.down_col(L, n, state=st)
            g, u = self.sm.in_rows(L, n, state=st)
            cs = torch.nn.functional.cosine_similarity
            weights["down_norm"][st] = float(dc_.norm())
            weights["down_cos_to_base"][st] = float(cs(dc_, d_base, dim=0))
            weights["gate_cos_to_base"][st] = float(cs(g, g0, dim=0))
            weights["up_cos_to_base"][st] = float(cs(u, u0, dim=0))
        unit = self.calib.act_per_unit(L, n)
        detail = {"id": rec["id"], "trajectory": traj, "by_domain": by_domain, "by_segment": by_segment,
                  "weights": weights, "lens": lens, "examples": uniq, "steering": [], "ablation": [],
                  "quantiles_base": qs,
                  "natural_range": {"p01": qs["p01"], "p99": qs["p99"], "resid_p50": self.calib.resid_p50[L],
                                    "down_norm": float(d_base.norm()), "add_unit_in_act": unit,
                                    "p99_in_resid_units": qs["p99"] / unit}}
        rec["base"]["q"] = {"p01": qs["p01"], "p50": qs["p5"], "p99": qs["p99"], "p999": qs["p999"]}
        return detail

    def run(self, candidates):
        out = self.args.run
        os.makedirs(os.path.join(out, "neurons", "detail"), exist_ok=True)
        os.makedirs(os.path.join(out, "neurons", "cards"), exist_ok=True)
        t0 = time.time()
        index = []
        for j, rec in enumerate(candidates):
            if rec["id"] not in self.neuron_k:
                print("no trace for", rec["id"])
                continue
            det = self.neuron(rec)
            json.dump(det, open(os.path.join(out, "neurons", "detail", rec["id"].replace("/", "_") + ".json"), "w"))
            cp = os.path.join(out, "neurons", "cards", rec["id"].replace("/", "_") + ".json")
            card = json.load(open(cp)) if os.path.exists(cp) else {"id": rec["id"], "status": "untested", "versions": [],
                                                                     "tests": [], "disagreements": []}
            json.dump(card, open(cp, "w"), indent=1)
            rec = dict(rec)
            rec["card_status"] = card["status"]
            rec["label"] = card["versions"][-1]["label"] if card["versions"] else ""
            rec.setdefault("detailed", False)
            index.append(rec)
            if j % 10 == 0:
                print(f"neuron {j + 1}/{len(candidates)} {time.time() - t0:.0f}s", flush=True)
        json.dump(index, open(os.path.join(out, "neurons", "index.json"), "w"), indent=1)
        self.manifest()
        self.corpus_index()

    def manifest(self):
        out = self.args.run
        adapters = {}
        for k, v in ADAPTER_DIRS.items():
            cfg = self.sm.adapter_cfg[k]
            adapters[k] = {"repo": "persona-cartography/monorepo", "revision": "507053d05ea65cd5c5023b07de1d4a0f10b13971",
                           "path": os.path.realpath(os.path.join(self.args.adapters, k)).split("monorepo/")[-1],
                           "r": cfg["r"], "alpha": cfg["lora_alpha"], "dropout": cfg["lora_dropout"],
                           "targets": sorted(cfg["target_modules"]), "base_model_in_config": cfg["base_model_name_or_path"]}
        man = {"run_id": os.path.basename(out.rstrip("/")), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "model": {"id": self.args.model, "revision": self.args.revision, "arch": self.sm.arch()},
               "adapters": adapters,
               "states": [{"id": s, "label": STATE_LABELS.get(s, s),
                           "trait": "openness" if s.startswith("o_") else "neuroticism" if s.startswith("n_") else None}
                          for s in self.states],
               "traits": {t: {"amp": a, "sup": p, "sweep": [f"{a}-1", f"{a}-0.5", "base", f"{a}+0.5", f"{a}+1"],
                              "poles": [f"{a}+1", f"{p}+1"]} for t, (a, p) in TRAIT_ADAPTERS.items()},
               "control_state": "ctrl+1",
               "corpus": {"n_docs": len(self.docs), "n_tokens": int(self.lengths.sum()), "domains": DOMAINS},
               "calibration": {"reference": "pn.intervene.REFERENCE_TEXT, base state", "resid_p50": self.calib.resid_p50},
               "notes": self.args.notes}
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)

    def corpus_index(self):
        os.makedirs(os.path.join(self.args.run, "corpus"), exist_ok=True)
        with open(os.path.join(self.args.run, "corpus", "docs.jsonl"), "w") as f:
            for d in self.docs:
                f.write(json.dumps({"doc_id": d["doc_id"], "source": d["source"], "domain": d["domain"],
                                    "n_tokens": len(d["ids"]), "boundary": d.get("boundary"),
                                    "split": split_of(d["doc_id"])}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--adapters", default="/workspace/adapters")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gate-up", action="store_true", default=True)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    b = Builder(args)
    b.run(json.load(open(args.candidates)))


if __name__ == "__main__":
    main()
