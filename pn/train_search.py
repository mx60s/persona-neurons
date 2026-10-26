"""Training-example search for shortlisted neurons (attribution levels 1 and 2 of the plan only).

    python -m pn.train_search --run runs/v1 --train-corpus data/train_corpus.jsonl --traces /workspace/out/train_traces_v1

For each neuron in runs/<id>/neurons/index.json with a trace:
  * windows from released training replies that most strongly activate it in the base state ("the example engages
    this response"), split by teacher (amplifying vs suppressing), and windows with the largest change under the
    trait's amplifier and suppressor ("the example exposes a learned difference"), appended to the neuron's detail
    examples with kind "training" and train_meta;
  * `training_selectivity`: mean activation over reply tokens on amplifying-teacher vs suppressing-teacher text, per
    state, in raw units and in units of the neuron's fixed denominator. A base-state difference is selectivity the
    base model already had; a change in the difference under an adapter is a changed use of it. Neither says which
    examples caused anything.
"""
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

PRE, POST = 28, 8
TRAIT_STATES = {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train-corpus", default="data/train_corpus.jsonl")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--tok", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tok, revision=args.revision)
    docs = [json.loads(l) for l in open(args.train_corpus)]
    meta = json.load(open(os.path.join(args.traces, "meta.json")))
    S = meta["states"]
    act = np.load(os.path.join(args.traces, "act.npy"), mmap_mode="r")
    kidx = {f"L{l}/N{n}": k for k, (l, n) in enumerate(meta["neurons"])}
    off = np.array(meta["offsets"])
    T = int(off[-1] + meta["lengths"][-1])
    doc_of = np.zeros(T, np.int32)
    pos_of = np.zeros(T, np.int32)
    reply = np.zeros(T, bool)
    for i, (o, n) in enumerate(zip(off, meta["lengths"])):
        doc_of[o:o + n] = i
        pos_of[o:o + n] = np.arange(n)
        reply[o + docs[i]["boundary"]:o + n] = True
    trait_of = np.array([docs[i]["trait"] for i in doc_of])
    teacher_of = np.array([docs[i]["teacher"] for i in doc_of])
    piece = {}

    def tstr(t):
        if t not in piece:
            piece[t] = tok.decode([t])
        return piece[t]

    def window(k, f, kind_note, states):
        i = int(doc_of[f])
        d = docs[i]
        p = int(pos_of[f])
        a, b = max(1, p - PRE), min(len(d["ids"]), p + POST + 1)
        o = int(off[i])
        return {"doc_id": d["doc_id"], "domain": "training", "source": d["source"], "kind": "training",
                "split": "training", "tokens": [tstr(t) for t in d["ids"][a:b]], "start": a, "focus": p - a,
                "boundary": (d["boundary"] - a) if a < d["boundary"] < b else None,
                "boundary_before_window": d["boundary"] <= a,
                "acts": {st: np.round(act[S.index(st), o + a:o + b, k].astype(np.float32), 4).tolist() for st in states},
                "train_meta": {"trait": d["trait"], "teacher": d["teacher"], "role_for_amplifier": d["role_for_amp"],
                               "role_for_suppressor": "rejected" if d["role_for_amp"] == "chosen" else "chosen",
                               "train_index": d["train_index"], "selected_as": kind_note}}

    index = json.load(open(os.path.join(args.run, "neurons", "index.json")))
    for rec in index:
        if rec["id"] not in kidx:
            continue
        k = kidx[rec["id"]]
        dp = os.path.join(args.run, "neurons", "detail", rec["id"].replace("/", "_") + ".json")
        det = json.load(open(dp))
        det["examples"] = [e for e in det["examples"] if e["kind"] != "training"]
        A = {st: act[S.index(st), :, k].astype(np.float32) for st in S}
        sel = {}
        for trait, (amp, sup) in TRAIT_STATES.items():
            m = reply & (trait_of == trait)
            hi, lo = m & (teacher_of == "amplifying"), m & (teacher_of == "suppressing")
            sel[trait] = {st: {"amp_teacher_mean": float(A[st][hi].mean()), "sup_teacher_mean": float(A[st][lo].mean()),
                               "diff": float(A[st][hi].mean() - A[st][lo].mean()),
                               "diff_z": float((A[st][hi].mean() - A[st][lo].mean()) / rec["denominator"])}
                          for st in ("base", amp, sup, "ctrl+1")}
        det["training_selectivity"] = sel
        traits = [t for t in TRAIT_STATES if t in rec.get("selected_for", [])] or list(TRAIT_STATES)
        shown = ["base"] + [s for t in TRAIT_STATES for s in TRAIT_STATES[t]] + ["ctrl+1"]
        new, used = [], set()

        def take(score, n, note):
            got = 0
            for f in np.argsort(-score):
                if not np.isfinite(score[f]):
                    break
                i = int(doc_of[f])
                if i in used:
                    continue
                used.add(i)
                new.append(window(k, int(f), note, shown))
                got += 1
                if got >= n:
                    break
        for t in traits:
            amp, sup = TRAIT_STATES[t]
            m = reply & (trait_of == t)
            for teacher in ("amplifying", "suppressing"):
                take(np.where(m & (teacher_of == teacher), A["base"], -np.inf), 2, f"top base activation, {teacher} teacher")
            take(np.where(m, np.abs(A[amp] - A["base"]), -np.inf), 3, f"largest change under {amp}")
            take(np.where(m, np.abs(A[sup] - A["base"]), -np.inf), 2, f"largest change under {sup}")
        for j, e in enumerate(new):
            e["ex_id"] = f"{rec['id'].replace('/', '_')}:t{j:02d}"
        det["examples"] += new
        json.dump(det, open(dp, "w"))
    print("added training examples for", sum(r["id"] in kidx for r in index), "neurons")


if __name__ == "__main__":
    main()
