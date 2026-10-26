"""Free-running behavioral generations: one short greedy answer per prompt per model state.

    python -m pn.behavior --prompts data/behavior_prompts.json --out runs/v1/behavior/responses.jsonl [--smoke]

Each trait's prompts run in that trait's seven conditions (base, amplifier at -1, -0.5, +0.5, +1, independent
suppressor at +1, control at +1). Prompts marked common also run in the other trait's conditions. Non-thinking chat
template, greedy decoding, max 160 new tokens, left-padded batches. Existing (prompt, state, intervention) rows are
skipped so the run can resume.
"""
import argparse
import json
import os
import time

import torch

from pn.collect import ADAPTER_DIRS
from pn.intervene import describe, intervene
from pn.states import StateModel

TRAIT_STATES = {
    "openness": ["base", "o_amp-1", "o_amp-0.5", "o_amp+0.5", "o_amp+1", "o_sup+1", "ctrl+1"],
    "neuroticism": ["base", "n_amp-1", "n_amp-0.5", "n_amp+0.5", "n_amp+1", "n_sup+1", "ctrl+1"],
}
# prompts not written for one trait (e.g. the earlier GRPO project's behavioral prompts) run in every state
TRAIT_STATES["general"] = sorted(set(TRAIT_STATES["openness"]) | set(TRAIT_STATES["neuroticism"]))
DECODING = {"do_sample": False, "max_new_tokens": 160}


def plan_jobs(prompts, smoke=False):
    """[(prompt, state)] in the order they will run, grouped by state."""
    jobs = {}
    for p in prompts:
        if smoke and not p.get("smoke"):
            continue
        states = set(TRAIT_STATES[p["trait"]])
        if p.get("common") and not smoke:
            for t in TRAIT_STATES:
                states |= set(TRAIT_STATES[t])
        for s in states:
            jobs.setdefault(s, []).append(p)
    return jobs


def generate(sm, prompts, state, edits=None, batch=32, max_new_tokens=160):
    """Greedy generations for a list of prompt dicts in one model state, optionally with a neuron intervention.
    Returns [(text, n_tokens, finish)]."""
    sm.set_state(state)
    tok = sm.tok
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    out = []
    for k in range(0, len(prompts), batch):
        chunk = prompts[k:k + batch]
        seqs = [sm.chat_ids(p["text"]) for p in chunk]
        T = max(map(len, seqs))
        ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=sm.device)
        mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=sm.device)
        ctx = intervene(sm, edits) if edits else _null()
        with ctx, torch.inference_mode():
            g = sm.model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                                  eos_token_id=eos, pad_token_id=pad, temperature=None, top_p=None, top_k=None)
        for row in g[:, T:].tolist():
            n, finish = len(row), "length"
            for i, t in enumerate(row):
                if t in eos:
                    n, finish = i, "eos"
                    break
            out.append((tok.decode(row[:n], skip_special_tokens=True), n, finish))
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/behavior_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--adapters", default="/workspace/adapters")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    prompts = [p for path in args.prompts.split(",") for p in json.load(open(path))]
    adapters = {k: os.path.join(args.adapters, k if os.path.isdir(os.path.join(args.adapters, k)) else v)
                for k, v in ADAPTER_DIRS.items()}
    sm = StateModel(args.model, adapters, revision=args.revision, device=args.device)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["prompt_id"], r["state"], json.dumps(r.get("intervention"))))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jobs = plan_jobs(prompts, args.smoke)
    t0 = time.time()
    with open(args.out, "a") as f:
        for state, ps in sorted(jobs.items()):
            ps = [p for p in ps if (p["id"], state, "null") not in done]
            if not ps:
                continue
            res = generate(sm, ps, state, batch=args.batch, max_new_tokens=args.max_new_tokens)
            for p, (text, n, finish) in zip(ps, res):
                rec = {"prompt_id": p["id"], **describe(state, None), "text": text, "n_tokens": n, "finish": finish,
                       "seed": 0, "decoding": {**DECODING, "max_new_tokens": args.max_new_tokens}}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{state}: {len(ps)} prompts, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
