"""Per-neuron effect metrics from pn.collect stats, and descriptive candidate groups.

    python -m pn.select metrics --stats /workspace/out/stats_v1 --adapters /workspace/adapters --out /workspace/out/metrics_v1.npz
    python -m pn.select pick --metrics /workspace/out/metrics_v1.npz --out /workspace/out/pick_v1

Standardization: every change for a neuron is divided by one denominator, the neuron's base-state RMS activation,
floored at the 10th percentile of base RMS within its layer, so tiny base activity cannot inflate effects. States are
never normalized independently. Raw values are kept alongside.

Metrics per state s and neuron (relative to base on identical tokens):
    mean_d, rms_d, z_mean, z_rms; agree (fraction of documents whose mean change has the overall sign);
    ci_lo/ci_hi (90% document-bootstrap interval of mean_d); rms between any two states (pair_rms);
    corr_amp_sup per trait (token-level correlation of the amplifier's and suppressor's changes);
    weight changes of the effective gate row, up row and down column (cosine to base, norm ratio);
    residual-contribution change along the base down column and its RMS.
"""
import argparse
import json
import os

import numpy as np
import torch

from pn.states import parse_state

TRAITS = {"openness": ("o_amp+1", "o_sup+1"), "neuroticism": ("n_amp+1", "n_sup+1")}
CONTROL = "ctrl+1"


def _lora_tensors(adapter_dir):
    from safetensors.torch import load_file
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    out = {}
    for k, v in sd.items():
        # base_model.model.model.layers.{i}.mlp.{proj}.lora_{A,B}.weight
        parts = k.split(".")
        if "mlp" not in parts:
            continue
        i = int(parts[parts.index("layers") + 1])
        proj = parts[parts.index("mlp") + 1]
        ab = "A" if "lora_A" in k else "B"
        out[(i, proj, ab)] = v
    return out, scale


def weight_metrics(model_path, adapters, states, L, device="cuda"):
    """Cosine to base and norm ratio of each neuron's effective gate row, up row and down column, per state."""
    from safetensors import safe_open
    ip = os.path.join(model_path, "model.safetensors.index.json")
    idx = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None
    lora = {k: _lora_tensors(v) for k, v in adapters.items()}
    out = {f"{p}_{m}": np.ones((len(states), L, 0), np.float32) for p in ("gate", "up", "down") for m in ("cos", "ratio")}
    res = {k: [] for k in out}
    base_norms = {"down": []}
    for i in range(L):
        W = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{i}.mlp.{proj}.weight"
            fn = idx[name] if idx else "model.safetensors"
            with safe_open(os.path.join(model_path, fn), "pt", device=device) as f:
                W[proj] = f.get_tensor(name).float()
        per = {k: [] for k in out}
        base_norms["down"].append(W["down_proj"].norm(dim=0).cpu().numpy())
        for st in states:
            spec = parse_state(st)
            for proj, key, dim in (("gate_proj", "gate", 1), ("up_proj", "up", 1), ("down_proj", "down", 0)):
                W0 = W[proj]
                dW = torch.zeros_like(W0)
                for a, c in spec.items():
                    t, s = lora[a]
                    dW += c * s * (t[(i, proj, "B")].to(device).float() @ t[(i, proj, "A")].to(device).float())
                We = W0 + dW
                cos = torch.nn.functional.cosine_similarity(We, W0, dim=dim)
                ratio = We.norm(dim=dim) / W0.norm(dim=dim).clamp(min=1e-8)
                per[f"{key}_cos"].append(cos.cpu().numpy())
                per[f"{key}_ratio"].append(ratio.cpu().numpy())
        for k in per:
            res[k].append(np.stack(per[k]))  # [S, N]
        print(f"weights layer {i}", flush=True)
    return {k: np.stack(v, 1).astype(np.float32) for k, v in res.items()}, np.stack(base_norms["down"])


def metrics(args):
    z = np.load(args.stats + ".npz")
    meta = json.load(open(args.stats + ".meta.json"))
    states = list(z["states"])
    S, L, N = z["sum_a"].shape
    n = z["n_tok"][:, None, None]
    mean_a, rms_a = z["sum_a"] / n, np.sqrt(z["sum_a2"] / n)
    mean_d, rms_d = z["sum_d"] / n, np.sqrt(np.maximum(z["sum_d2"] / n, 0))
    floor = np.percentile(rms_a[0], 10, axis=1, keepdims=True)
    denom = np.maximum(rms_a[0], np.maximum(floor, 1e-4))
    C = z["cross"]
    pair_rms = np.zeros((S, S, L, N), np.float32)
    for i in range(S):
        for j in range(i):
            v = (C[i, i] + C[j, j] - 2 * C[i, j]) / n[0]
            pair_rms[i, j] = pair_rms[j, i] = np.sqrt(np.maximum(v, 0))
    corr = {}
    for t, (A, P) in TRAITS.items():
        if A in states and P in states:
            a, p = states.index(A), states.index(P)
            sdd = (C[max(a, p), min(a, p)] - C[a, 0] - C[p, 0] + C[0, 0]) / n[0]
            cov = sdd - mean_d[a] * mean_d[p]
            va = rms_d[a] ** 2 - mean_d[a] ** 2
            vp = rms_d[p] ** 2 - mean_d[p] ** 2
            corr[t] = (cov / np.sqrt(np.maximum(va * vp, 1e-20))).astype(np.float32)

    # document-level agreement and bootstrap
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dm = np.load(args.stats + ".doc_mean_d.npy", mmap_mode="r")  # [S, D, L, N] float16
    doc_n = torch.tensor(z["doc_n"], device=dev, dtype=torch.float32)
    D = doc_n.numel()
    g = torch.Generator(device="cpu").manual_seed(0)
    B = args.boot
    counts = torch.stack([torch.bincount(torch.randint(0, D, (D,), generator=g), minlength=D) for _ in range(B)]).float().to(dev)
    w = counts * doc_n[None]  # token-weighted bootstrap of documents
    agree = np.zeros((S, L, N), np.float32)
    ci_lo, ci_hi = np.zeros((S, L, N), np.float32), np.zeros((S, L, N), np.float32)
    for s in range(1, S):
        for l in range(L):
            x = torch.from_numpy(np.asarray(dm[s, :, l])).to(dev).float()  # [D, N]
            overall = torch.from_numpy(mean_d[s, l]).to(dev).float()
            sgn = torch.sign(overall)
            agree[s, l] = ((torch.sign(x) == sgn[None]).float().mean(0)).cpu().numpy()
            bs = (w @ x) / w.sum(1, keepdim=True)  # [B, N]
            q = torch.quantile(bs, torch.tensor([0.05, 0.95], device=dev), dim=0)
            ci_lo[s, l], ci_hi[s, l] = q[0].cpu().numpy(), q[1].cpu().numpy()
        print(f"bootstrap state {states[s]}", flush=True)

    adapters = {k: os.path.join(args.adapters, k) for k in ("o_amp", "o_sup", "n_amp", "n_sup", "ctrl")}
    from huggingface_hub import snapshot_download
    mp = meta["model"] if os.path.isdir(meta["model"]) else snapshot_download(meta["model"], revision=meta["revision"])
    wm, down_norm = weight_metrics(mp, adapters, states, L, dev)
    # residual contribution: a * d. Along the base direction: mean(a_s) * (d_s . d_b_hat) - mean(a_b) * |d_b|
    # = |d_b| * (mean(a_s) * cos_s * ratio_s - mean(a_b)); RMS of the full change vector uses the cross terms.
    cos_s, ratio_s = wm["down_cos"], wm["down_ratio"]
    resid_mean_d = down_norm[None] * (mean_a * cos_s * ratio_s - mean_a[0][None])
    resid_rms_d = np.zeros_like(resid_mean_d)
    for s in range(1, S):
        v = (C[s, s] * ratio_s[s] ** 2 - 2 * C[s, 0] * ratio_s[s] * cos_s[s] + C[0, 0]) / n[0]
        resid_rms_d[s] = down_norm * np.sqrt(np.maximum(v, 0))
    np.savez(args.out, states=np.array(states), n_tok=z["n_tok"], mean_a=mean_a.astype(np.float32),
             rms_a=rms_a.astype(np.float32), max_a=z["max_a"], min_a=z["min_a"],
             frac_pos=(z["n_pos"] / n).astype(np.float32), frac_neg=(z["n_neg"] / n).astype(np.float32),
             mean_d=mean_d.astype(np.float32), rms_d=rms_d.astype(np.float32), denom=denom.astype(np.float32),
             z_mean=(mean_d / denom).astype(np.float32), z_rms=(rms_d / denom).astype(np.float32),
             agree=agree, ci_lo=ci_lo, ci_hi=ci_hi, pair_rms=pair_rms,
             **{f"corr_{t}": v for t, v in corr.items()}, down_norm=down_norm.astype(np.float32),
             resid_mean_d=resid_mean_d.astype(np.float32), resid_rms_d=resid_rms_d.astype(np.float32),
             grp_mean_d=(z["grp_sum_d"] / np.maximum(z["grp_n"], 1)[None, :, None, None]).astype(np.float32),
             grp_n=z["grp_n"], **wm)
    print("wrote", args.out)


def pick(args):
    m = np.load(args.metrics)
    states = list(m["states"])
    zr, zm, agree = m["z_rms"], m["z_mean"], m["agree"]
    S, L, N = zr.shape
    c = states.index(CONTROL)
    pair = m["pair_rms"] / m["denom"][None, None]
    rng = np.random.default_rng(0)
    chosen, table = {}, []

    def add(t, group, flat_ids, k, reason):
        got = 0
        for f in flat_ids:
            l, nn = divmod(int(f), N)
            key = f"L{l}/N{nn}"
            if key in chosen and t in chosen[key]["selected_for"]:
                continue
            rec = chosen.setdefault(key, {"id": key, "layer": l, "n": nn, "kind": "candidate", "selected_for": [],
                                          "group": {}, "rank": {}, "selection_reason": []})
            rec["selected_for"].append(t)
            rec["group"][t] = group
            rec["rank"][t] = got + 1
            rec["selection_reason"].append(f"{t}/{group}: {reason}")
            got += 1
            if got >= k:
                break

    quota = {"shared": args.shared, "amp_primary": args.primary, "sup_primary": args.primary}
    for t, (A, P) in TRAITS.items():
        a, p = states.index(A), states.index(P)
        eA, eP, eC = zr[a], zr[p], zr[c]
        corr = m[f"corr_{t}"]
        # trait-specific magnitude: change in the trait state relative to the matched control state
        sA, sP = pair[a, c], pair[p, c]
        active = (m["rms_a"][0] > 0) & (m["denom"] > 0)
        # shared: both poles change the neuron, beyond the control
        shared = np.minimum(np.minimum(eA, sA), np.minimum(eP, sP))
        shared = np.where(active, shared, -1).ravel()
        add(t, "shared", np.argsort(-shared), quota["shared"], "large change under both poles, beyond control")
        # primarily one pole: ratio test plus control excess
        ampp = np.where(active & (eP < 0.5 * eA) & (eC < 0.5 * eA), np.minimum(eA, sA) - eP, -1).ravel()
        add(t, "amp_primary", np.argsort(-ampp), quota["amp_primary"], "amplifier change at least 2x suppressor and control")
        supp = np.where(active & (eA < 0.5 * eP) & (eC < 0.5 * eP), np.minimum(eP, sP) - eA, -1).ravel()
        add(t, "sup_primary", np.argsort(-supp), quota["sup_primary"], "suppressor change at least 2x amplifier and control")

    # cross-trait: large trait-specific change under both traits (any pole)
    o = np.maximum(np.minimum(zr[states.index("o_amp+1")], pair[states.index("o_amp+1"), c]),
                   np.minimum(zr[states.index("o_sup+1")], pair[states.index("o_sup+1"), c]))
    ne = np.maximum(np.minimum(zr[states.index("n_amp+1")], pair[states.index("n_amp+1"), c]),
                    np.minimum(zr[states.index("n_sup+1")], pair[states.index("n_sup+1"), c]))
    cross = np.minimum(o, ne).ravel()
    add("cross", "cross_trait", np.argsort(-cross), args.cross, "large change under openness and neuroticism adapters")
    # control-associated: the neutral adapter changes it as much as any trait adapter
    trait_max = np.max(np.stack([zr[states.index(s)] for s in ("o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1")]), 0)
    ctrl_assoc = np.where(zr[c] >= trait_max, zr[c], -1).ravel()
    add("control", "control_assoc", np.argsort(-ctrl_assoc), args.control, "neutral control changes it at least as much as any trait adapter")

    cand = list(chosen.values())
    # layer-matched random controls
    layers = [r["layer"] for r in cand if r["kind"] == "candidate"]
    taken = {(r["layer"], r["n"]) for r in cand}
    ctrls = []
    for l in rng.choice(layers, size=min(args.random, len(layers)), replace=False):
        while True:
            nn = int(rng.integers(N))
            if (l, nn) not in taken:
                break
        taken.add((int(l), nn))
        ctrls.append({"id": f"L{l}/N{nn}", "layer": int(l), "n": nn, "kind": "random_control", "selected_for": [],
                      "group": {}, "rank": {}, "selection_reason": ["layer-matched random control"]})
    allr = cand + ctrls
    for r in allr:
        l, nn = r["layer"], r["n"]
        r["selection_reason"] = "; ".join(r["selection_reason"])
        r["base"] = {"mean": float(m["mean_a"][0, l, nn]), "rms": float(m["rms_a"][0, l, nn]),
                     "frac_pos": float(m["frac_pos"][0, l, nn]), "frac_neg": float(m["frac_neg"][0, l, nn]),
                     "max": float(m["max_a"][0, l, nn]), "min": float(m["min_a"][0, l, nn])}
        r["denominator"] = float(m["denom"][l, nn])
        r["effects"] = {}
        for s, st in enumerate(states):
            if s == 0:
                continue
            r["effects"][st] = {"mean_diff": float(m["mean_d"][s, l, nn]), "rms_diff": float(m["rms_d"][s, l, nn]),
                                "z_mean": float(zm[s, l, nn]), "z_rms": float(zr[s, l, nn]),
                                "agree": float(agree[s, l, nn]), "ci90": [float(m["ci_lo"][s, l, nn]), float(m["ci_hi"][s, l, nn])],
                                "z_vs_control": float(pair[s, c, l, nn]),
                                "resid_mean_diff": float(m["resid_mean_d"][s, l, nn]),
                                "resid_rms_diff": float(m["resid_rms_d"][s, l, nn]),
                                "down_cos": float(m["down_cos"][s, l, nn]), "gate_cos": float(m["gate_cos"][s, l, nn]),
                                "up_cos": float(m["up_cos"][s, l, nn])}
        r["corr_amp_sup"] = {t: float(m[f"corr_{t}"][l, nn]) for t in TRAITS}
        # within-layer percentile of each trait-pole effect, for comparison with random neurons
        r["layer_pct"] = {st: float((zr[states.index(st), l] < zr[states.index(st), l, nn]).mean())
                          for st in ("o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", CONTROL)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(allr, open(os.path.join(args.out, "candidates.json"), "w"), indent=1)
    json.dump([[r["layer"], r["n"]] for r in allr], open(os.path.join(args.out, "neurons.json"), "w"))
    # scatter data for figures: amp vs sup effect for every neuron (float16), per trait, plus control
    np.savez_compressed(os.path.join(args.out, "scatter.npz"),
                        **{f"{t}_{k}": zr[states.index(s)].astype(np.float16) for t, (A, P) in TRAITS.items()
                           for k, s in (("amp", A), ("sup", P))}, control=zr[c].astype(np.float16),
                        **{f"corr_{t}": m[f"corr_{t}"].astype(np.float16) for t in TRAITS})
    n_c = sum(r["kind"] == "candidate" for r in allr)
    print(f"{n_c} candidates, {len(ctrls)} random controls")
    for t in list(TRAITS) + ["cross", "control"]:
        print(t, sum(t in r["selected_for"] for r in allr))
    multi = [r["id"] for r in allr if len(r["selected_for"]) > 1]
    print("selected under more than one heading:", len(multi), multi[:20])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("metrics")
    a.add_argument("--stats", required=True)
    a.add_argument("--adapters", default="/workspace/adapters")
    a.add_argument("--out", required=True)
    a.add_argument("--boot", type=int, default=200)
    b = sub.add_parser("pick")
    b.add_argument("--metrics", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--shared", type=int, default=10)
    b.add_argument("--primary", type=int, default=9)
    b.add_argument("--cross", type=int, default=6)
    b.add_argument("--control", type=int, default=3)
    b.add_argument("--random", type=int, default=12)
    args = ap.parse_args()
    {"metrics": metrics, "pick": pick}[args.cmd](args)


if __name__ == "__main__":
    main()
