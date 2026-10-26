"""Weight-space comparison of the adapters' effective updates, per layer and module.

    python -m pn.weights --adapters /workspace/adapters --out runs/v1/weights.json

For every layer and target module: the Frobenius norm of each adapter's effective update dW = alpha/r * B @ A, and the
cosine between pairs of updates, computed exactly from the factors: <B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T)).
"""
import argparse
import json
import os
from itertools import combinations

import torch
from safetensors.torch import load_file

NAMES = ["o_amp", "o_sup", "n_amp", "n_sup", "ctrl"]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(adir):
    sd = load_file(os.path.join(adir, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
    s = cfg["lora_alpha"] / cfg["r"]
    out = {}
    for k, v in sd.items():
        parts = k.split(".")
        i = int(parts[parts.index("layers") + 1])
        mod = next(m for m in MODULES if m in parts)
        out.setdefault((i, mod), {})["A" if "lora_A" in k else "B"] = v.double()
    return out, s


def inner(f1, s1, f2, s2):
    return s1 * s2 * float(((f1["B"].T @ f2["B"]) * (f1["A"] @ f2["A"].T)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="/workspace/adapters")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ad = {n: load(os.path.join(args.adapters, n)) for n in NAMES}
    layers = sorted({i for i, _ in ad["o_amp"][0]})
    rows = []
    for i in layers:
        for mod in MODULES:
            f = {n: ad[n][0][(i, mod)] for n in NAMES}
            s = {n: ad[n][1] for n in NAMES}
            norm = {n: inner(f[n], s[n], f[n], s[n]) ** 0.5 for n in NAMES}
            cos = {f"{a}|{b}": inner(f[a], s[a], f[b], s[b]) / (norm[a] * norm[b]) for a, b in combinations(NAMES, 2)}
            rows.append({"layer": i, "module": mod, "norm": norm, "cos": cos})
    # whole-adapter cosines (all modules and layers stacked)
    tot = {}
    for a, b in combinations(NAMES, 2):
        num = sum(inner(ad[a][0][k], ad[a][1], ad[b][0][k], ad[b][1]) for k in ad[a][0])
        na = sum(inner(ad[a][0][k], ad[a][1], ad[a][0][k], ad[a][1]) for k in ad[a][0]) ** 0.5
        nb = sum(inner(ad[b][0][k], ad[b][1], ad[b][0][k], ad[b][1]) for k in ad[b][0]) ** 0.5
        tot[f"{a}|{b}"] = num / (na * nb)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"per_module": rows, "total_cos": tot}, open(args.out, "w"), indent=1)
    for k, v in tot.items():
        print(f"{k:14s} cos {v:+.3f}")


if __name__ == "__main__":
    main()
