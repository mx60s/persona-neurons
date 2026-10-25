"""Model states: the base Qwen3 model plus any combination of scaled personality adapters, one copy in memory.

All five adapters are loaded once into a single PeftModel. A state is a dict {adapter_name: coefficient}; the empty
dict is the base model. Setting a state activates exactly those adapters and sets each LoRA layer's scaling to
coefficient * lora_alpha / r, so the forward pass adds the sum of the adapters' effective weight updates
(coefficient * alpha/r * B @ A). Only the effective update is scaled, never both factors (which would square the
coefficient). The base state disables the adapter layers entirely, so it runs the untouched base weights.

State ids are strings: "base", "o_amp+1", "o_amp-0.5", "n_sup+1", "ctrl+1", "o_amp+1|n_sup+0.5".
"""
import json
import os
import re
from contextlib import contextmanager

import torch

STATE_RE = re.compile(r"^([a-z_]+)([+-]\d+(?:\.\d+)?)$")


def parse_state(state_id):
    """'o_amp+1|n_sup-0.5' -> {'o_amp': 1.0, 'n_sup': -0.5}; 'base' -> {}."""
    if state_id in ("base", "", None):
        return {}
    out = {}
    for part in state_id.split("|"):
        m = STATE_RE.match(part.strip())
        if not m:
            raise ValueError(f"bad state id part {part!r} in {state_id!r}")
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def fmt_coef(c):
    s = f"{c:+g}"
    return s


def state_id(spec):
    spec = {k: v for k, v in spec.items() if v != 0}
    if not spec:
        return "base"
    return "|".join(f"{k}{fmt_coef(v)}" for k, v in sorted(spec.items()))


class StateModel:
    """Base model + named adapters with switchable scaled states.

    model_id/revision: base checkpoint. adapters: {name: local adapter dir}. Use .lm for the decoder stack
    (embed_tokens, layers, norm), .model for the full causal LM (forward, lm_head) and .tok for the tokenizer.
    """

    def __init__(self, model_id, adapters, revision=None, device=None, dtype=torch.bfloat16, tokenizer_id=None):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.dtype = device, dtype
        self.model_id, self.revision = model_id, revision
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id or model_id, revision=None if tokenizer_id else revision)
        base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).to(device).eval()
        self.adapter_cfg = {}
        names = list(adapters)
        if names:
            pm = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0], is_trainable=False)
            for name in names[1:]:
                pm.load_adapter(adapters[name], adapter_name=name, is_trainable=False)
            for name in names:
                cfg = json.load(open(os.path.join(adapters[name], "adapter_config.json")))
                self.adapter_cfg[name] = cfg
            self.peft = pm.eval()
            self.model = pm.base_model.model  # the Qwen3ForCausalLM whose linears are now LoRA wrappers
        else:
            self.peft = None
            self.model = base
        self.lm = self.model.model
        self.names = names
        self._lora_layers = [m for m in self.model.modules() if _is_lora_layer(m)]
        self.state = None
        self.set_state("base")

    # ---------- states ----------
    def set_state(self, state):
        """state: state id string or {adapter: coefficient}. Returns the canonical state id."""
        spec = parse_state(state) if isinstance(state, str) else dict(state)
        spec = {k: float(v) for k, v in spec.items() if float(v) != 0.0}
        for k in spec:
            if k not in self.adapter_cfg:
                raise KeyError(f"unknown adapter {k!r}; have {self.names}")
        sid = state_id(spec)
        if sid == self.state:
            return sid
        if not spec:
            for m in self._lora_layers:
                m.enable_adapters(False)
        else:
            active = sorted(spec)
            for m in self._lora_layers:
                m.enable_adapters(True)
                m.set_adapter(active)
                for k in active:
                    if k in m.scaling:
                        cfg = self.adapter_cfg[k]
                        r = m.r[k]
                        alpha = m.lora_alpha[k]
                        if cfg.get("use_rslora"):
                            base_scale = alpha / (r ** 0.5)
                        else:
                            base_scale = alpha / r
                        m.scaling[k] = spec[k] * base_scale
                for p in m.parameters():
                    p.requires_grad_(False)
        self.state = sid
        return sid

    @contextmanager
    def in_state(self, state):
        prev = self.state
        try:
            yield self.set_state(state)
        finally:
            self.set_state(prev)

    # ---------- effective weights ----------
    def effective_weight(self, layer, proj, state=None):
        """Effective weight of mlp.{proj} (gate_proj, up_proj, down_proj) at `layer` under `state` (default current):
        W + sum_k c_k * alpha_k/r_k * B_k @ A_k, as float32."""
        mod = getattr(self.lm.layers[layer].mlp, proj)
        W = _base_weight(mod).float()
        spec = parse_state(state if state is not None else self.state)
        if _is_lora_layer(mod):
            for k, c in spec.items():
                if k in mod.lora_A:
                    s = mod.lora_alpha[k] / (mod.r[k] ** 0.5 if self.adapter_cfg[k].get("use_rslora") else mod.r[k])
                    W = W + c * s * (mod.lora_B[k].weight.float() @ mod.lora_A[k].weight.float())
        return W

    def down_col(self, layer, n, state=None):
        """Column n of the effective down projection: the direction neuron n writes into the residual stream."""
        mod = self.lm.layers[layer].mlp.down_proj
        col = _base_weight(mod)[:, n].float()
        spec = parse_state(state if state is not None else self.state)
        if _is_lora_layer(mod):
            for k, c in spec.items():
                if k in mod.lora_A:
                    s = mod.lora_alpha[k] / (mod.r[k] ** 0.5 if self.adapter_cfg[k].get("use_rslora") else mod.r[k])
                    col = col + c * s * (mod.lora_B[k].weight.float() @ mod.lora_A[k].weight[:, n].float())
        return col

    def in_rows(self, layer, n, state=None):
        """Rows n of the effective gate and up projections (what neuron n reads), float32 [hidden] each."""
        out = []
        for proj in ("gate_proj", "up_proj"):
            mod = getattr(self.lm.layers[layer].mlp, proj)
            row = _base_weight(mod)[n].float()
            spec = parse_state(state if state is not None else self.state)
            if _is_lora_layer(mod):
                for k, c in spec.items():
                    if k in mod.lora_A:
                        s = mod.lora_alpha[k] / (mod.r[k] ** 0.5 if self.adapter_cfg[k].get("use_rslora") else mod.r[k])
                        row = row + c * s * (mod.lora_B[k].weight[n].float() @ mod.lora_A[k].weight.float())
            out.append(row)
        return out

    def down_proj_module(self, layer):
        """The module whose input is the neuron activation vector silu(gate) * up (LoRA wrapper when adapters exist).
        A forward pre-hook here sees, and can rewrite, the activations fed to both base and adapter paths."""
        return self.lm.layers[layer].mlp.down_proj

    @property
    def n_layers(self):
        return len(self.lm.layers)

    @property
    def n_inter(self):
        return _base_weight(self.lm.layers[0].mlp.down_proj).shape[1]

    def arch(self):
        c = self.model.config
        return {"model": self.model_id, "revision": self.revision, "n_layers": c.num_hidden_layers,
                "hidden": c.hidden_size, "inter": c.intermediate_size, "attn_heads": c.num_attention_heads,
                "kv_heads": c.num_key_value_heads, "head_dim": getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
                "act": c.hidden_act, "model_type": c.model_type, "adapters": self.names}

    # ---------- chat formatting ----------
    def chat_ids(self, messages, add_generation_prompt=True):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                           enable_thinking=False, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)


def _is_lora_layer(m):
    return hasattr(m, "lora_A") and hasattr(m, "scaling") and hasattr(m, "base_layer")


def _base_weight(mod):
    return mod.base_layer.weight if _is_lora_layer(mod) else mod.weight
