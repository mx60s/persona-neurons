"""Live Qwen3 + adapter states for the explorer, on top of pn.states.StateModel (one model, switchable states).

What it does, all within one model state at a time:
- reads one neuron's per-token activation on a text, for a list of states (same tokens in every state);
- greedy chat generation in a state, optionally with a neuron intervention per batch row;
- next-token distribution with and without a neuron intervention;
- weight-derived views (logit lens of the down column, direct-path connections) for a given state.

The hook is a forward pre-hook on each layer's mlp.down_proj (the PEFT LoRA wrapper when adapters are loaded), whose
input is the activation vector silu(gate) * up. Reading and rewriting there is exact, and a rewrite feeds both the
base and the adapter paths of down_proj.

Add doses are in "resid_p50" units, calibrated against a FIXED base-state reference: the median residual-stream norm
leaving the layer on REFERENCE_TEXT, measured in the base state, divided by the neuron's base down-column norm. So
"add +1" adds resid_p50 / |d_base| to the activation, whichever model state is active. The same activation change
writes a slightly different residual vector in an adapted state when the adapter changed the down column; the
calibration endpoint reports both norms.
"""
import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pn.states import StateModel, parse_state, state_id  # noqa: E402

ADAPTER_NAMES = ["o_amp", "o_sup", "n_amp", "n_sup", "ctrl"]

# Lyra's fixed calibration text (mixed prose, code, dialogue); fed raw, no chat template
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)

MAX_NEW_CAP = 400
MAX_READ_TOKENS = 2048


def parse_neuron(nid):
    """'L17/N4021' -> (17, 4021)."""
    s = nid.strip().upper().replace(" ", "")
    if not s.startswith("L") or "/N" not in s:
        raise ValueError(f"bad neuron id {nid!r}, expected L<layer>/N<index>")
    a, b = s[1:].split("/N")
    return int(a), int(b)


def canon_state(s):
    return state_id(parse_state(s))


class Live:
    def __init__(self, model_id, adapters_dir=None, revision=None, device=None, dtype="bf16", tokenizer_id=None):
        self.status = "loading"
        self.error = None
        adapters = {}
        if adapters_dir:
            for name in ADAPTER_NAMES:
                p = os.path.join(adapters_dir, name)
                if os.path.exists(os.path.join(p, "adapter_config.json")):
                    adapters[name] = p
        self.adapters = adapters
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.sm = StateModel(model_id, adapters, revision=revision, device=device, dtype=dt, tokenizer_id=tokenizer_id)
        self.tok = self.sm.tok
        self.dev = self.sm.device
        self.lock = threading.RLock()
        self._read = None  # (layer, n) while reading
        self._seen = {}
        self._push = None  # [(layer, n, mode, per-row tensor)]
        self._want_resid = False
        self._resid = {}
        self._calib = None  # {layer: resid_p50} in base state on REFERENCE_TEXT
        self._lens_cache = {}
        self._con_cache = {}
        for i, layer in enumerate(self.sm.lm.layers):
            self.sm.down_proj_module(i).register_forward_pre_hook(self._hook(i))
            layer.register_forward_hook(self._resid_hook(i))
        self.status = "ready"

    # ---------- hooks ----------
    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self._read is not None and self._read[0] == i:
                self._seen["act"] = h[0, :, self._read[1]].float().cpu()
            mine = [p for p in self._push if p[0] == i] if self._push else []
            if not mine:
                return None
            h = h.clone()
            for _, n, mode, vals in mine:
                v = vals.to(h.dtype)
                if mode == "mul":
                    h[:, :, n] = h[:, :, n] * v[:, None]
                else:  # add (zero is mul by 0)
                    h[:, :, n] = h[:, :, n] + v[:, None]
            return (h,) + tuple(inp[1:])
        return hook

    def _resid_hook(self, i):
        def hook(mod, inp, out):
            if self._want_resid:
                h = out[0] if isinstance(out, tuple) else out
                self._resid[i] = h[0, 1:].float().norm(dim=-1).cpu()
        return hook

    # ---------- metadata ----------
    def info(self):
        a = self.sm.arch()
        a.update(device=str(self.dev), dtype=str(self.sm.dtype).replace("torch.", ""),
                 adapters={k: {"path": v, "r": self.sm.adapter_cfg[k].get("r"),
                               "alpha": self.sm.adapter_cfg[k].get("lora_alpha"),
                               "dropout": self.sm.adapter_cfg[k].get("lora_dropout"),
                               "targets": self.sm.adapter_cfg[k].get("target_modules"),
                               "base": self.sm.adapter_cfg[k].get("base_model_name_or_path")}
                           for k, v in self.adapters.items()})
        return a

    def check_neuron(self, layer, n):
        if not (0 <= layer < self.sm.n_layers and 0 <= n < self.sm.n_inter):
            raise ValueError(f"L{layer}/N{n} is outside this model ({self.sm.n_layers} layers x {self.sm.n_inter})")

    def check_state(self, s):
        spec = parse_state(s)
        missing = [k for k in spec if k not in self.adapters]
        if missing:
            raise ValueError(f"state {s!r} needs adapters not loaded: {missing}")
        return state_id(spec)

    # ---------- calibration (fixed base reference) ----------
    def resid_p50(self, layer):
        with self.lock:
            if self._calib is None:
                ids = self.tok(REFERENCE_TEXT).input_ids
                with self.sm.in_state("base"), torch.inference_mode():
                    self._want_resid, self._resid = True, {}
                    try:
                        self.sm.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
                    finally:
                        self._want_resid = False
                self._calib = {i: float(v.median()) for i, v in self._resid.items()}
                self._calib_tokens = len(ids)
            return self._calib[layer]

    def calibration(self, layer, n, state="base"):
        """Units for add doses. Fixed: base-state resid_p50 on the reference text over the base down-column norm."""
        self.check_neuron(layer, n)
        state = self.check_state(state)
        r = self.resid_p50(layer)
        with self.lock:
            d_base = float(self.sm.down_col(layer, n, "base").norm())
            d_state = float(self.sm.down_col(layer, n, state).norm())
        return {"layer": layer, "n": n, "resid_p50": r, "down_norm_base": d_base, "unit_act": r / d_base,
                "state": state, "down_norm_state": d_state, "reference": "base state, fixed reference text",
                "reference_tokens": self._calib_tokens,
                "note": "add +1 = activation + resid_p50 / down_norm_base, in every model state"}

    def _push_entry(self, layer, n, iv_rows):
        """iv_rows: per batch row an intervention dict or None. Returns hook pushes, grouped by (layer, n, mode)."""
        groups = {}
        for r, iv in enumerate(iv_rows):
            if not iv or iv.get("mode", "none") == "none":
                continue
            mode = iv["mode"]
            if mode == "add":
                key, val = "add", float(iv["value"]) * self.resid_p50(layer) / float(self.sm.down_col(layer, n, "base").norm())
            elif mode == "mul":
                key, val = "mul", float(iv["value"])
            elif mode == "zero":
                key, val = "mul", 0.0
            else:
                raise ValueError(f"unknown intervention mode {mode!r}")
            if key not in groups:
                groups[key] = [0.0 if key == "add" else 1.0] * len(iv_rows)
            groups[key][r] = val
        return [(layer, n, k, torch.tensor(v, device=self.dev, dtype=torch.float32)) for k, v in groups.items()] or None

    # ---------- token display units ----------
    def token_units(self, ids):
        """Lyra's grouping: byte pieces of one multi-byte character join the token that completes it.
        Returns [(text, [token indices])]."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        return units

    def encode(self, text, chat):
        if chat:
            ids = self.sm.chat_ids(text)
        else:
            ids = self.tok(text).input_ids
        if len(ids) > MAX_READ_TOKENS:
            raise ValueError(f"text is {len(ids)} tokens; limit {MAX_READ_TOKENS}")
        return ids

    # ---------- reading ----------
    def read(self, layer, n, text, states, chat=False):
        """Per-token activation of neuron n at `layer` on the same token ids in each state. No intervention."""
        self.check_neuron(layer, n)
        states = list(dict.fromkeys(self.check_state(s) for s in states))
        ids = self.encode(text, chat)
        acts = {}
        with self.lock:
            x = torch.tensor([ids], device=self.dev)
            for s in states:
                with self.sm.in_state(s), torch.inference_mode():
                    self._read = (layer, n)
                    try:
                        self.sm.lm(input_ids=x, use_cache=False)
                    finally:
                        self._read = None
                    acts[s] = self._seen.pop("act").tolist()
        units = self.token_units(ids)
        out_units = [{"text": t, "idx": idx} for t, idx in units]
        unit_acts = {s: [max((a[i] for i in idx), key=abs) for _, idx in units] for s, a in acts.items()}
        return {"tokens": [u["text"] for u in out_units], "acts": unit_acts, "n_tokens": len(ids), "chat": chat,
                "states": states}

    # ---------- generation ----------
    def generate(self, state, rows_messages, iv_rows, layer=None, n=None, max_new=128, cancel=None):
        """Greedy-decodes len(rows_messages) conversations in one batch in model `state`; row r gets intervention
        iv_rows[r] on neuron (layer, n) at every position. Yields (row, text_delta), then (None, None)."""
        max_new = max(1, min(int(max_new), MAX_NEW_CAP))
        state = self.check_state(state)
        if layer is not None:
            self.check_neuron(layer, n)
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            push = self._push_entry(layer, n, iv_rows) if layer is not None else None
            seqs = [self.sm.chat_ids(m) for m in rows_messages]
            B, T = len(seqs), max(map(len, seqs))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
            mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            stop = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
            done, out_ids, sent = [False] * B, [[] for _ in range(B)], [""] * B
            try:
                with self.sm.in_state(state), torch.inference_mode():
                    self._push = push
                    out = self.sm.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                                        logits_to_keep=1)
                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            break
                        nxt = out.logits[:, -1].argmax(-1)
                        for r in range(B):
                            if done[r]:
                                continue
                            t = int(nxt[r])
                            if t in stop:
                                done[r] = True
                                continue
                            out_ids[r].append(t)
                            text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, text[len(sent[r]):]
                                sent[r] = text
                        if all(done):
                            break
                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = self.sm.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                            past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                self._push = None
        yield None, None

    # ---------- next-token distribution ----------
    def next_token(self, state, layer, n, text, ivs, chat=True, prefix="", k=10):
        """Next-token log-probs after `text` (+ `prefix`) in `state`, unmodified and under each intervention in
        `ivs`. Returns top tokens of the unmodified distribution and, per intervention, KL(unmod || intervened),
        the top tokens, and the largest log-prob increases and decreases."""
        self.check_neuron(layer, n)
        state = self.check_state(state)
        ids = self.encode(text, chat)
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids
        rows = [None] + list(ivs)
        with self.lock:
            push = self._push_entry(layer, n, rows)
            x = torch.tensor([ids] * len(rows), device=self.dev)
            try:
                with self.sm.in_state(state), torch.inference_mode():
                    self._push = push
                    out = self.sm.model(input_ids=x, use_cache=False, logits_to_keep=1)
            finally:
                self._push = None
            lp = torch.log_softmax(out.logits[:, -1].float(), -1).cpu()

        def dec(t):
            return self.tok.decode([t])

        def top(row):
            v, i = row.topk(k)
            return [[dec(int(t)), round(float(p.exp()), 4)] for p, t in zip(v, i)]

        base = lp[0]
        res = {"state": state, "n_tokens": len(ids), "unmodified": {"top": top(base)}, "variants": []}
        for r, iv in enumerate(ivs, start=1):
            d = lp[r] - base
            kl = float((base.exp() * (base - lp[r])).sum())
            # rank changes among tokens in the top 200 of either distribution
            relevant = torch.zeros_like(d, dtype=torch.bool)
            relevant[base.topk(200).indices] = True
            relevant[lp[r].topk(200).indices] = True
            dd = torch.where(relevant, d, torch.zeros_like(d))
            up_v, up_i = dd.topk(k)
            dn_v, dn_i = (-dd).topk(k)
            res["variants"].append({
                "intervention": iv, "kl": round(kl, 5), "top": top(lp[r]),
                "up": [[dec(int(t)), round(float(v), 3), round(float(lp[r][t].exp()), 4)] for v, t in zip(up_v, up_i) if v > 0],
                "down": [[dec(int(t)), round(-float(v), 3), round(float(lp[r][t].exp()), 4)] for v, t in zip(dn_v, dn_i) if v > 0],
            })
        return res

    # ---------- weight-derived views, per state ----------
    def lens(self, layer, n, state, k=12):
        """Tokens the neuron's effective down column (in `state`) pushes up / down through the final norm and
        lm_head. A positive activation writes +d, a negative activation writes -d."""
        self.check_neuron(layer, n)
        state = self.check_state(state)
        key = (layer, n, state, k)
        if key in self._lens_cache:
            return self._lens_cache[key]
        with self.lock, torch.inference_mode():
            d = self.sm.down_col(layer, n, state).to(self.sm.dtype)
            D = torch.stack([d, -d])[:, None]  # [2, 1, hidden]
            logits = self.sm.model.lm_head(self.sm.lm.norm(D))[:, 0].float()
            out = {"state": state,
                   "up": [self.tok.decode([t]) for t in logits[0].topk(k).indices.tolist()],
                   "down": [self.tok.decode([t]) for t in logits[1].topk(k).indices.tolist()]}
        self._lens_cache[key] = out
        return out

    def _eff_matvec(self, mod, v, spec):
        """Effective weight (base + scaled LoRA updates in spec) times vector v, without forming the matrix."""
        W = mod.base_layer.weight if hasattr(mod, "base_layer") else mod.weight
        out = (W @ v.to(W.dtype)).float()  # base part in the model dtype: no fp32 copy of a 12288 x 4096 matrix
        if hasattr(mod, "lora_A"):
            for kname, c in spec.items():
                if kname in mod.lora_A:
                    s = mod.lora_alpha[kname] / (mod.r[kname] ** 0.5 if self.sm.adapter_cfg[kname].get("use_rslora") else mod.r[kname])
                    out = out + c * s * (mod.lora_B[kname].weight.float() @ (mod.lora_A[kname].weight.float() @ v))
        return out

    def _eff_vecmat(self, mod, r, spec):
        """r^T times the effective weight of mod."""
        W = mod.base_layer.weight if hasattr(mod, "base_layer") else mod.weight
        out = (r.to(W.dtype) @ W).float()
        if hasattr(mod, "lora_A"):
            for kname, c in spec.items():
                if kname in mod.lora_A:
                    s = mod.lora_alpha[kname] / (mod.r[kname] ** 0.5 if self.sm.adapter_cfg[kname].get("use_rslora") else mod.r[kname])
                    out = out + c * s * ((r @ mod.lora_B[kname].weight.float()) @ mod.lora_A[kname].weight.float())
        return out

    def connections(self, layer, n, state, k=16):
        """Direct-path weights between this neuron and neurons in other layers, in `state`, ignoring everything in
        between (attention, other MLPs, the norm's input-dependent scale). Qwen3 RMSNorm multiplies by its weight w
        directly (not 1 + w as in Qwen3.5).
        Downstream B in layer j > layer: g = Wgate_j[B] . (w_j * d), u = Wup_j[B] . (w_j * d), d this neuron's
        effective down column. Upstream A in layer i < layer: A's down column against this neuron's gate/up rows
        (each times w of this layer). Score = |g * u| z-scored within each layer; sign = sign(g * u)."""
        self.check_neuron(layer, n)
        state = self.check_state(state)
        key = (layer, n, state, k)
        if key in self._con_cache:
            return self._con_cache[key]
        spec = parse_state(state)
        L = self.sm.lm.layers
        out = {"state": state, "down": [], "up": []}
        with self.lock, torch.inference_mode():
            d = self.sm.down_col(layer, n, state)
            g_row, u_row = self.sm.in_rows(layer, n, state)
            w_me = L[layer].post_attention_layernorm.weight.float()
            r_g, r_u = g_row * w_me, u_row * w_me
            for j in range(len(L)):
                if j == layer:
                    continue
                if j > layer:
                    v = d * L[j].post_attention_layernorm.weight.float()
                    g = self._eff_matvec(L[j].mlp.gate_proj, v, spec)
                    u = self._eff_matvec(L[j].mlp.up_proj, v, spec)
                    side = "down"
                else:
                    g = self._eff_vecmat(L[j].mlp.down_proj, r_g, spec)
                    u = self._eff_vecmat(L[j].mlp.down_proj, r_u, spec)
                    side = "up"
                prod = g * u
                mag = prod.abs()
                z = (mag - mag.mean()) / (mag.std() + 1e-8)
                top = z.topk(min(k, z.numel()))
                out[side] += [{"id": f"L{j}/N{int(i)}", "layer": j, "n": int(i), "z": round(float(sc), 2),
                               "sign": 1 if prod[i] > 0 else -1}
                              for sc, i in zip(top.values.tolist(), top.indices.tolist())]
        for side in ("down", "up"):
            out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
        if len(self._con_cache) > 500:
            self._con_cache.clear()
        self._con_cache[key] = out
        return out
