"""Checks pn.collect stats and traces against a brute-force recomputation (each document alone, no padding).

    python -m pn.collect stats --corpus /tmp/tiny_corpus.jsonl --model ../work/tiny/base --revision main \
        --adapters ../work/tiny/adapters --out /tmp/tiny_stats --max-tokens 512 --device cpu
    python -m pn.collect traces ... --neurons /tmp/tiny_neurons.json --out /tmp/tiny_traces
    python tests/test_collect.py /tmp/tiny_corpus.jsonl /tmp/tiny_stats /tmp/tiny_traces
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pn.collect import DOMAINS  # noqa: E402
from pn.states import StateModel  # noqa: E402

corpus, stats, traces = sys.argv[1:4]
docs = [json.loads(l) for l in open(corpus)]
z = np.load(stats + ".npz")
states = list(z["states"])
names = ["o_amp", "o_sup", "n_amp", "n_sup", "ctrl"]
sm = StateModel("../work/tiny/base", {n: f"../work/tiny/adapters/{n}" for n in names}, device="cpu", dtype=torch.float32)
L, N = sm.n_layers, sm.n_inter
acts = {}  # state -> list of [L, T, N] per doc
cap = {}
hs = [sm.down_proj_module(i).register_forward_pre_hook(lambda m, inp, i=i: cap.__setitem__(i, inp[0][0].float()))
      for i in range(L)]
for st in states:
    sm.set_state(st)
    acts[st] = []
    for d in docs:
        with torch.inference_mode():
            sm.lm(input_ids=torch.tensor([d["ids"]]))
        acts[st].append(torch.stack([cap[i] for i in range(L)]))
for h in hs:
    h.remove()

ok = True


def check(name, a, b, tol=2e-3):
    global ok
    err = float(np.abs(np.asarray(a) - np.asarray(b)).max() / (np.abs(np.asarray(b)).max() + 1e-9))
    good = err < tol
    ok &= good
    print(("PASS " if good else "FAIL ") + name, f"relmax={err:.2e}")


for s, st in enumerate(states):
    A = torch.cat([x[:, 1:] for x in acts[st]], 1)  # skip position 0
    B = torch.cat([x[:, 1:] for x in acts["base"]], 1)
    check(f"{st} sum_a", z["sum_a"][s], A.sum(1).numpy())
    check(f"{st} sum_a2", z["sum_a2"][s], (A * A).sum(1).numpy())
    check(f"{st} max_a", z["max_a"][s], A.max(1).values.numpy())
    check(f"{st} n_pos", z["n_pos"][s], (A > 0).sum(1).numpy(), tol=0.02)
    if s:
        D = A - B
        check(f"{st} sum_d", z["sum_d"][s], D.sum(1).numpy())
        check(f"{st} sum_d2", z["sum_d2"][s], (D * D).sum(1).numpy())
        for j in range(s):
            Aj = torch.cat([x[:, 1:] for x in acts[states[j]]], 1)
            check(f"{st} cross {states[j]}", z["cross"][s, j], (A * Aj).sum(1).numpy())
        dm = np.load(stats + ".doc_mean_d.npy")[s]
        ref = np.stack([(acts[st][k][:, 1:] - acts["base"][k][:, 1:]).mean(1).numpy() for k in range(len(docs))])
        check(f"{st} doc_mean_d", dm, ref, tol=5e-3)
        g = np.zeros((len(DOMAINS) * 3, L, N))
        for k, d in enumerate(docs):
            seg = np.full(len(d["ids"]), 2) if d["boundary"] is None else (np.arange(len(d["ids"])) >= d["boundary"]).astype(int)
            gid = DOMAINS.index(d["domain"]) * 3 + seg[1:]
            Dk = (acts[st][k][:, 1:] - acts["base"][k][:, 1:]).numpy()
            for t, gg in enumerate(gid):
                g[gg] += Dk[:, t]
        check(f"{st} grp_sum_d", z["grp_sum_d"][s], g)

if os.path.exists(os.path.join(traces, "meta.json")):
    meta = json.load(open(os.path.join(traces, "meta.json")))
    act = np.load(os.path.join(traces, "act.npy"))
    for s, st in enumerate(meta["states"]):
        sm_states = meta["states"]
        for k, (l, n) in enumerate(meta["neurons"]):
            ref = np.concatenate([acts[st][j][l, :, n].numpy() for j in range(len(docs))]) if st in acts else None
            if ref is not None:
                check(f"trace {st} L{l}/N{n}", act[s, :, k].astype(np.float32), ref, tol=5e-3)
print("ALL PASS" if ok else "SOME FAILED")
