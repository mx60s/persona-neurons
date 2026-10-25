"""Checks the adapter arithmetic the whole study depends on. Runs on the tiny fixture locally and on Qwen3-8B on the pod:

    python tests/test_states.py --base ../work/tiny/base --adapters ../work/tiny/adapters
    python tests/test_states.py --base Qwen/Qwen3-8B --revision <sha> --adapters /workspace/adapters --dtype bf16

1. base state == untouched base model (separately loaded, no PEFT) on the same input;
2. coefficient 0 == base; disable / re-enable gives back identical logits;
3. a scaled, summed state == a model whose weights were merged by hand with W + sum c * alpha/r * B @ A (tiny only:
   needs a second copy of the weights);
4. single adapter at +1 == PEFT's own default forward for that adapter;
5. down_col / in_rows agree with effective_weight.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pn.states import StateModel, parse_state  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--revision", default=None)
ap.add_argument("--adapters", required=True, help="dir holding o_amp, o_sup, n_amp, n_sup, ctrl")
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--skip-merge", action="store_true")
args = ap.parse_args()

dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
names = ["o_amp", "o_sup", "n_amp", "n_sup", "ctrl"]
adapters = {n: os.path.join(args.adapters, n) for n in names}
sm = StateModel(args.base, adapters, revision=args.revision, dtype=dtype)
tol = 1e-4 if dtype == torch.float32 else 2e-2
ids = torch.tensor([sm.chat_ids("Tell me something surprising about octopuses.")], device=sm.device)


def logits(model=None):
    with torch.inference_mode():
        return (model or sm.model)(input_ids=ids).logits.float()


def rel(a, b):
    return float((a - b).norm() / b.norm())


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, detail)


sm.set_state("base")
l_base = logits()
from transformers import AutoModelForCausalLM  # noqa: E402

if not args.skip_merge:
    plain = AutoModelForCausalLM.from_pretrained(args.base, revision=args.revision, dtype=dtype).to(sm.device).eval()
    l_plain = logits(plain)
    check("base state == plain base model", rel(l_base, l_plain) < tol / 10, f"rel={rel(l_base, l_plain):.2e}")
    del plain
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

sm.set_state({"o_amp": 0.0})
check("coef 0 == base", sm.state == "base" and rel(logits(), l_base) == 0)

sm.set_state("o_amp+1")
l_o1 = logits()
check("o_amp+1 differs from base", rel(l_o1, l_base) > 1e-3, f"rel={rel(l_o1, l_base):.2e}")
sm.set_state("base")
check("disable after enable == base", rel(logits(), l_base) == 0)
sm.set_state("o_amp+1")
check("re-enable reproduces o_amp+1", rel(logits(), l_o1) == 0)

# PEFT's own forward with its default scaling for a single adapter
sm.peft.set_adapter("o_amp")
for m in sm._lora_layers:
    m.enable_adapters(True)
    m.set_adapter("o_amp")
    m.scaling["o_amp"] = m.lora_alpha["o_amp"] / m.r["o_amp"]
sm.state = None
check("o_amp+1 == PEFT default forward", rel(logits(), l_o1) < 1e-6)

sm.set_state("o_amp-1")
l_om1 = logits()
sm.set_state("o_amp+0.5")
l_oh = logits()
check("-1 and +0.5 are distinct states", rel(l_om1, l_o1) > 1e-3 and rel(l_oh, l_o1) > 1e-4)

if not args.skip_merge:
    spec = parse_state("o_amp-0.5|n_sup+1|ctrl+1")
    sm.set_state(spec)
    l_combo = logits()
    merged = AutoModelForCausalLM.from_pretrained(args.base, revision=args.revision, dtype=dtype).to(sm.device).eval()
    with torch.no_grad():
        for li in range(sm.n_layers):
            for parent, projs in (("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                                  ("mlp", ["gate_proj", "up_proj", "down_proj"])):
                for p in projs:
                    lw = getattr(getattr(sm.lm.layers[li], parent), p)
                    tgt = getattr(getattr(merged.model.layers[li], parent), p)
                    W = tgt.weight.float()
                    for k, c in spec.items():
                        s = lw.lora_alpha[k] / lw.r[k]
                        W = W + c * s * (lw.lora_B[k].weight.float() @ lw.lora_A[k].weight.float())
                    tgt.weight.copy_(W.to(dtype))
    l_merged = logits(merged)
    check("summed scaled state == hand-merged weights", rel(l_combo, l_merged) < tol, f"rel={rel(l_combo, l_merged):.2e}")
    # the squared-coefficient mistake would be visibly different
    del merged

sm.set_state("n_amp-0.5|o_sup+1")
W = sm.effective_weight(1, "down_proj")
check("down_col matches effective_weight", torch.allclose(sm.down_col(1, 3), W[:, 3], atol=1e-5))
g, u = sm.in_rows(1, 5)
check("in_rows matches effective_weight",
      torch.allclose(g, sm.effective_weight(1, "gate_proj")[5], atol=1e-5)
      and torch.allclose(u, sm.effective_weight(1, "up_proj")[5], atol=1e-5))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
