"""Live measurement of neuron activations on arbitrary texts, and numerical checking of card predictions.

    python -m pn.measure check --run runs/v1 [--neurons L17/N4021,...] [--state base]

Texts are measured as plain text (no chat template). Position 0 is the attention sink and is reported as 0 so it can
never count as a hit. Results are appended to each card's tests by pn.explain.check_card.
"""
import argparse
import json
import os

import torch

from pn.collect import ADAPTER_DIRS
from pn.states import StateModel


class Measurer:
    def __init__(self, model="Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218",
                 adapters="/workspace/adapters", device=None):
        ad = {k: os.path.join(adapters, k if os.path.isdir(os.path.join(adapters, k)) else v)
              for k, v in ADAPTER_DIRS.items()}
        self.sm = StateModel(model, ad, revision=revision, device=device)
        self.calls = 0

    def acts(self, texts, neuron, state="base", chat=False):
        layer, n = int(neuron.split("/")[0][1:]), int(neuron.split("/")[1][1:])
        sm = self.sm
        sm.set_state(state)
        cap = {}
        h = sm.down_proj_module(layer).register_forward_pre_hook(
            lambda m, inp: cap.__setitem__("a", inp[0][0, :, n].float().cpu()))
        out = []
        try:
            for t in texts:
                ids = sm.chat_ids(t) if chat else sm.tok(t).input_ids
                with torch.inference_mode():
                    sm.lm(input_ids=torch.tensor([ids], device=sm.device), use_cache=False)
                a = cap["a"].tolist()
                a[0] = 0.0
                out.append({"tokens": [sm.tok.decode([i]) for i in ids], "acts": a})
                self.calls += 1
        finally:
            h.remove()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--neurons", default="all")
    ap.add_argument("--state", default="base")
    ap.add_argument("--version", type=int, default=None)
    args = ap.parse_args()
    from pn import explain
    m = Measurer()
    cards_dir = os.path.join(args.run, "neurons", "cards")
    ids = [f[:-5].replace("_", "/") for f in sorted(os.listdir(cards_dir))] if args.neurons == "all" else args.neurons.split(",")
    for nid in ids:
        card = json.load(open(os.path.join(cards_dir, nid.replace("/", "_") + ".json")))
        if not card.get("versions"):
            continue
        ent = explain.check_card(args.run, nid, lambda texts, neuron, state: m.acts(texts, neuron, state),
                                 state=args.state, version=args.version, source=f"live Qwen3-8B, state {args.state}")
        res = [e["result"] for e in ent]
        print(nid, {r: res.count(r) for r in set(res)}, "->", json.load(open(os.path.join(cards_dir, nid.replace("/", "_") + ".json")))["status"], flush=True)


if __name__ == "__main__":
    main()
