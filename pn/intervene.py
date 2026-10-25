"""Neuron interventions within a model state, separate from the state itself.

An intervention edits selected activations (inputs to down_proj) during the forward pass: add a value, multiply, or
zero. Add doses can be given in raw activation units ("act") or in Lyra's residual-relative units ("resid_p50"): one
unit moves the residual stream by the base model's median residual norm at that layer along the neuron's base down
column, i.e. act_delta = dose * resid_p50_base[layer] / |d_base|. The reference is always the base model on a fixed
reference text, so a dose means the same activation change in every model state.
"""
from contextlib import contextmanager

import torch

REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


class Calibration:
    """Base-state residual norms on REFERENCE_TEXT, measured once per model, for residual-relative doses."""

    def __init__(self, sm):
        self.sm = sm
        self.resid_p50 = None

    def measure(self):
        sm = self.sm
        ids = torch.tensor([sm.tok(REFERENCE_TEXT).input_ids], device=sm.device)
        norms = {}
        hs = [layer.register_forward_hook(lambda m, i, o, li=li: norms.__setitem__(
            li, (o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).median().item()))
            for li, layer in enumerate(sm.lm.layers)]
        try:
            with sm.in_state("base"), torch.inference_mode():
                sm.lm(input_ids=ids, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        self.resid_p50 = [norms[i] for i in range(sm.n_layers)]
        return self.resid_p50

    def act_per_unit(self, layer, n):
        """Activation change corresponding to one residual-relative unit for neuron (layer, n)."""
        if self.resid_p50 is None:
            self.measure()
        return self.resid_p50[layer] / float(self.sm.down_col(layer, n, state="base").norm())


@contextmanager
def intervene(sm, edits, rows=None, positions=None):
    """edits: [(layer, n, mode, value)] with mode in add / mul / zero and value in raw activation units (add) or a
    factor (mul). rows: optional per-batch-row multipliers for the edit sign, e.g. (1, 0, -1) gives Lyra's
    positive / unmodified / negative columns in one batch (for mul, row 0 uses value, row 1 leaves it, row 2 uses
    -value). positions: optional slice of sequence positions to edit (default all)."""
    by_layer = {}
    for layer, n, mode, value in edits:
        by_layer.setdefault(layer, []).append((n, mode, value))
    handles = []

    def make(layer):
        def hook(mod, inp):
            h = inp[0].clone()
            sl = positions if positions is not None else slice(None)
            for n, mode, value in by_layer[layer]:
                if rows is None:
                    r = torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
                else:
                    r = torch.tensor(rows, device=h.device, dtype=h.dtype)
                if mode == "add":
                    h[:, sl, n] = h[:, sl, n] + (r * value)[:, None]
                elif mode == "mul":
                    f = torch.where(r > 0, torch.full_like(r, value), torch.where(r < 0, torch.full_like(r, -value),
                                                                                  torch.ones_like(r)))
                    h[:, sl, n] = h[:, sl, n] * f[:, None]
                elif mode == "zero":
                    keep = (r == 0).to(h.dtype) if rows is not None else torch.zeros_like(r)
                    h[:, sl, n] = h[:, sl, n] * keep[:, None]
                else:
                    raise ValueError(mode)
            return (h,)
        return hook

    try:
        for layer in by_layer:
            handles.append(sm.down_proj_module(layer).register_forward_pre_hook(make(layer)))
        yield
    finally:
        for h in handles:
            h.remove()


def describe(state, edits, unit="act", calib=None):
    """The record attached to every saved result: model state and neuron intervention, kept separate."""
    if not edits:
        return {"state": state, "intervention": None}
    out = []
    for layer, n, mode, value in edits:
        e = {"neuron": f"L{layer}/N{n}", "mode": mode, "value": value, "unit": "x" if mode == "mul" else unit}
        if mode == "add" and calib is not None:
            e["value_act"] = value * (calib.act_per_unit(layer, n) if unit == "resid_p50" else 1.0)
        out.append(e)
    return {"state": state, "intervention": out[0] if len(out) == 1 else out}
