"""Do the openness and neuroticism adapters push each neuron the same way or opposite ways?

    python3 scripts/analysis/cross_trait.py /workspace/out/traces_full data/corpus.jsonl runs/v2/cross_trait.json

For every traced neuron: token-level correlation between the change under each openness pole and each neuroticism
pole (change = adapted - base, same tokens), with all tokens and with chat-template (reply-start) tokens excluded, plus
each pole's change size relative to the neuron's base RMS.
"""
import json
import sys

import numpy as np

traces, corpus, out = sys.argv[1:4]
meta = json.load(open(f"{traces}/meta.json"))
S = meta["states"]
docs = [json.loads(l) for l in open(corpus)]
IM_START, FIRST_SPECIAL = 151644, 151643


def tmask(d):
    ids, b = d["ids"], d.get("boundary")
    chat = d["domain"] in ("chat", "emotional_chat")
    m = np.zeros(len(ids), bool)
    for i, t in enumerate(ids):
        if t >= FIRST_SPECIAL or (i > 0 and ids[i - 1] == IM_START) or (i > 1 and ids[i - 2] == IM_START):
            m[i] = True
        if chat and b is not None and b - 6 <= i < b:
            m[i] = True
    return m


valid = np.concatenate([np.r_[False, np.ones(len(d["ids"]) - 1, bool)] for d in docs])
text = valid & ~np.concatenate([tmask(d) for d in docs])
act = np.load(f"{traces}/act.npy", mmap_mode="r")
K = act.shape[2]
idx = {s: S.index(s) for s in ("base", "o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", "ctrl+1")}
PAIRS = [("o_amp+1", "n_amp+1"), ("o_amp+1", "n_sup+1"), ("o_sup+1", "n_amp+1"), ("o_sup+1", "n_sup+1")]
res = []
CH = 64
for k0 in range(0, K, CH):
    block = {s: np.asarray(act[i, :, k0:k0 + CH], dtype=np.float32) for s, i in idx.items()}  # [T, ch]
    for j in range(block["base"].shape[1]):
        l, n = meta["neurons"][k0 + j]
        b = block["base"][:, j]
        rb = float(np.sqrt((b[valid] ** 2).mean())) or 1e-6
        d = {s: block[s][:, j] - b for s in idx if s != "base"}
        rec = {"id": f"L{l}/N{n}", "z": {s: float(np.sqrt((d[s][valid] ** 2).mean()) / rb) for s in d}}
        for name, m in (("all", valid), ("text", text)):
            for a, c in PAIRS:
                x, y = d[a][m], d[c][m]
                rec[f"corr_{a}|{c}|{name}"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
        res.append(rec)
    print(f"{min(k0 + CH, K)}/{K}", flush=True)
json.dump(res, open(out, "w"))
print("wrote", out, len(res))
