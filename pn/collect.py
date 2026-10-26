"""Teacher-forced activation comparison across model states on the fixed corpus.

Activations are the inputs to each layer's down_proj (silu(gate) * up), read by a forward pre-hook on the down_proj
module, which is the LoRA wrapper when adapters are loaded, so they are exactly what the adapted model computes.
Every batch runs the base state first and keeps its activations; each other state then runs on the identical tokens
and its statistics against base are accumulated inside the hook. Position 0 of every document (the attention sink) is
excluded from statistics.

    # all neurons, streamed statistics
    python -m pn.collect stats --corpus data/corpus.jsonl --states base,o_amp+1,o_sup+1,n_amp+1,n_sup+1,ctrl+1 \
        --out /workspace/out/stats_v1 [--smoke]
    # selected neurons, per-token traces (+ gate and up)
    python -m pn.collect traces --corpus data/corpus.jsonl --neurons out/shortlist.json --states <all> --out ...

stats output (npz, float32 unless noted; S = states, L = layers, N = neurons, G = domain x segment groups,
D = documents):
    n_tok[S], sum_a/sum_a2/n_pos/n_neg/max_a/min_a [S, L, N], sum_d/sum_d2 [S, L, N] (d = a_state - a_base),
    cross [S, S, L, N] float64 (cross[s, j] = sum a_s * a_j, j <= s), grp_sum_a/grp_sum_d [S, G, L, N], grp_n[G], doc_mean_d [S, D, L, N] float16 (separate
    file), doc_n[D].
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from pn.states import StateModel

DOMAINS = ["chat", "emotional_chat", "emotional_text", "narrative", "factual", "web", "code", "training"]
SEGMENTS = ["prompt", "continuation", "nosplit"]
ADAPTER_DIRS = {
    "o_amp": "openness_amplifying_full-persona",
    "o_sup": "openness_suppressing_full-persona",
    "n_amp": "neuroticism_amplifying_full-persona",
    "n_sup": "neuroticism_suppressing_full-persona",
    "ctrl": "ocean_def_control_full-persona",
}


def load_corpus(path, smoke=False):
    docs = [json.loads(l) for l in open(path)]
    if smoke:
        docs = [d for d in docs if d.get("smoke")]
    return docs


def batches(docs, max_tokens):
    """Groups consecutive documents into batches of at most max_tokens padded positions."""
    cur, longest = [], 0
    for i, d in enumerate(docs):
        L = len(d["ids"])
        if cur and max(longest, L) * (len(cur) + 1) > max_tokens:
            yield cur
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        yield cur


IM_START, FIRST_SPECIAL = 151644, 151643


def template_mask(d):
    """Chat-template positions of a document: special tokens, the role token after <|im_start|> and the newline after
    it, and for chat documents the assistant header and empty think block just before the reply."""
    ids = d["ids"]
    m = [False] * len(ids)
    b = d.get("boundary")
    chat = d["domain"] in ("chat", "emotional_chat", "training")
    for i, t in enumerate(ids):
        if t >= FIRST_SPECIAL or (i > 0 and ids[i - 1] == IM_START) or (i > 1 and ids[i - 2] == IM_START):
            m[i] = True
        if chat and b is not None and b - 6 <= i < b:
            m[i] = True
    return m


def make_batch(docs, idx, pad_id, device, exclude_template=False):
    T = max(len(docs[i]["ids"]) for i in idx)
    ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), T), dtype=torch.long)
    group = torch.full((len(idx), T), -1, dtype=torch.long)
    for r, i in enumerate(idx):
        d = docs[i]
        n = len(d["ids"])
        ids[r, :n] = torch.tensor(d["ids"])
        mask[r, :n] = 1
        di = DOMAINS.index(d["domain"])
        b = d.get("boundary")
        seg = torch.full((n,), 2) if b is None else (torch.arange(n) >= b).long()
        group[r, :n] = di * 3 + seg
    valid = mask.bool().clone()
    valid[:, 0] = False  # attention sink
    if exclude_template:
        for r, i in enumerate(idx):
            tm = torch.tensor(template_mask(docs[i]))
            valid[r, :len(tm)] &= ~tm
    doc_of = torch.tensor(idx)[:, None].expand_as(ids)
    return ids.to(device), mask.to(device), valid.flatten().to(device), group.flatten().to(device), doc_of.flatten().to(device)


def load_model(args):
    adapters = {k: os.path.join(args.adapters, k if os.path.isdir(os.path.join(args.adapters, k)) else v)
                for k, v in ADAPTER_DIRS.items()}
    return StateModel(args.model, adapters, revision=args.revision, device=args.device,
                      dtype={'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype])


class StatsCollector:
    def __init__(self, sm, states, n_docs, device):
        L, N, S = sm.n_layers, sm.n_inter, len(states)
        G = len(DOMAINS) * 3
        self.sm, self.states, self.L, self.N, self.S, self.G = sm, states, L, N, S, G
        f = lambda *shape: torch.zeros(shape, dtype=torch.float64, device=device)  # noqa: E731
        self.n_tok = torch.zeros(S, dtype=torch.float64)
        self.sum_a, self.sum_a2, self.n_pos, self.n_neg = f(S, L, N), f(S, L, N), f(S, L, N), f(S, L, N)
        self.sum_d, self.sum_d2 = f(S, L, N), f(S, L, N)
        self.cross = f(S, S, L, N)  # cross[s, j] = sum a_s * a_j for j <= s (cross[s, s] = sum a_s^2)
        self.max_a = torch.full((S, L, N), -1e9, dtype=torch.float32, device=device)
        self.min_a = torch.full((S, L, N), 1e9, dtype=torch.float32, device=device)
        self.grp_sum_a = torch.zeros((S, G, L, N), dtype=torch.float32, device=device)
        self.grp_sum_d = torch.zeros((S, G, L, N), dtype=torch.float32, device=device)
        self.grp_n = torch.zeros(G, dtype=torch.float64)
        self.doc_sum_d = torch.zeros((S, n_docs, L, N), dtype=torch.float32, device=device)
        self.doc_n = torch.zeros(n_docs, dtype=torch.float64)
        self.cache = [[None] * L for _ in range(S)]  # this batch's activations per state (bf16)
        self.s = 0
        self.valid = self.group = self.doc = None
        self.handles = [sm.down_proj_module(i).register_forward_pre_hook(self._hook(i)) for i in range(L)]

    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            a = h.reshape(-1, h.shape[-1])[self.valid].float()  # [tokens, N]
            s = self.s
            self.cache[s][i] = h.reshape(-1, h.shape[-1])[self.valid]  # bf16, exactly what the model computed
            b = self.cache[0][i].float()
            self.sum_a[s, i] += a.sum(0)
            self.sum_a2[s, i] += (a * a).sum(0)
            self.n_pos[s, i] += (a > 0).sum(0)
            self.n_neg[s, i] += (a < 0).sum(0)
            self.cross[s, s, i] += (a * a).sum(0)
            self.max_a[s, i] = torch.maximum(self.max_a[s, i], a.max(0).values)
            self.min_a[s, i] = torch.minimum(self.min_a[s, i], a.min(0).values)
            self.grp_sum_a[s, :, i].index_add_(0, self.group, a)
            if s > 0:
                d = a - b
                self.sum_d[s, i] += d.sum(0)
                self.sum_d2[s, i] += (d * d).sum(0)
                for j in range(s):
                    self.cross[s, j, i] += (a * self.cache[j][i].float()).sum(0)
                self.grp_sum_d[s, :, i].index_add_(0, self.group, d)
                self.doc_sum_d[s, :, i].index_add_(0, self.doc, d)
            return None
        return hook

    def run_batch(self, ids, mask, valid, group, doc):
        self.valid = valid
        self.group = group[valid]
        self.doc = doc[valid]
        ntok = int(valid.sum())
        self.grp_n += torch.bincount(self.group, minlength=self.G).double().cpu()
        self.doc_n += torch.bincount(self.doc, minlength=self.doc_n.numel()).double().cpu()
        for s, st in enumerate(self.states):
            self.s = s
            self.sm.set_state(st)
            with torch.inference_mode():
                self.sm.lm(input_ids=ids, attention_mask=mask, use_cache=False)
            self.n_tok[s] += ntok
        self.cache = [[None] * self.L for _ in range(self.S)]

    def save(self, out):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        c = lambda t: t.float().cpu().numpy()  # noqa: E731
        np.savez(out + ".npz", states=np.array(self.states), domains=np.array(DOMAINS), segments=np.array(SEGMENTS),
                 n_tok=self.n_tok.numpy(), sum_a=c(self.sum_a), sum_a2=c(self.sum_a2), n_pos=c(self.n_pos),
                 n_neg=c(self.n_neg), max_a=c(self.max_a), min_a=c(self.min_a), sum_d=c(self.sum_d),
                 sum_d2=c(self.sum_d2), cross=self.cross.cpu().numpy(), grp_sum_a=c(self.grp_sum_a),
                 grp_sum_d=c(self.grp_sum_d), grp_n=self.grp_n.numpy(), doc_n=self.doc_n.numpy())
        dn = self.doc_n.clamp(min=1).to(self.doc_sum_d.device).float()[None, :, None, None]
        np.save(out + ".doc_mean_d.npy", (self.doc_sum_d / dn).half().cpu().numpy())


class TraceCollector:
    """Per-token activation, gate and up values of selected neurons in every state, into memory-mapped arrays
    [S, total_tokens, K] (float16). Token order is corpus order; positions include the attention sink."""

    def __init__(self, sm, states, neurons, total_tokens, out):
        self.sm, self.states = sm, states
        self.neurons = neurons  # [(layer, n)]
        self.K = len(neurons)
        self.by_layer = {}
        for k, (l, n) in enumerate(neurons):
            self.by_layer.setdefault(l, ([], []))
            self.by_layer[l][0].append(k)
            self.by_layer[l][1].append(n)
        os.makedirs(out, exist_ok=True)
        shape = (len(states), total_tokens, self.K)
        self.mm = {name: np.lib.format.open_memmap(os.path.join(out, f"{name}.npy"), mode="w+", dtype=np.float16,
                                                   shape=shape) for name in ("act", "gate", "up")}
        self.buf = {}
        self.handles = []
        for l, (ks, ns) in self.by_layer.items():
            mlp = sm.lm.layers[l].mlp
            idx = torch.tensor(ns, device=sm.device)
            self.handles.append(mlp.down_proj.register_forward_pre_hook(self._pre(l, idx)))
            self.handles.append(mlp.gate_proj.register_forward_hook(self._post(l, idx, "gate")))
            self.handles.append(mlp.up_proj.register_forward_hook(self._post(l, idx, "up")))

    def _pre(self, l, idx):
        def hook(mod, inp):
            self.buf[("act", l)] = inp[0][..., idx].float()
        return hook

    def _post(self, l, idx, name):
        def hook(mod, inp, out):
            self.buf[(name, l)] = out[..., idx].float()
        return hook

    def run_batch(self, ids, mask, offsets_lengths):
        for s, st in enumerate(self.states):
            self.sm.set_state(st)
            self.buf.clear()
            with torch.inference_mode():
                self.sm.lm(input_ids=ids, attention_mask=mask, use_cache=False)
            for name in ("act", "gate", "up"):
                full = torch.zeros((ids.shape[0], ids.shape[1], self.K), device=ids.device)
                for l, (ks, _) in self.by_layer.items():
                    full[..., ks] = self.buf[(name, l)]
                full = full.half().cpu().numpy()
                for r, (off, n) in enumerate(offsets_lengths):
                    self.mm[name][s, off:off + n] = full[r, :n]

    def close(self):
        for m in self.mm.values():
            m.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats", "traces", "bench"])
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--adapters", default="/workspace/adapters")
    ap.add_argument("--states", default="base,o_amp+1,o_sup+1,n_amp+1,n_sup+1,ctrl+1")
    ap.add_argument("--neurons", help="json list of [layer, n] (traces mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exclude-template", action="store_true", help="leave chat-template positions out of the statistics")
    ap.add_argument("--max-tokens", type=int, default=2048, help="padded positions per batch")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--limit-batches", type=int, default=0)
    args = ap.parse_args()

    states = args.states.split(",")
    assert states[0] == "base", "first state must be base"
    docs = load_corpus(args.corpus, args.smoke)
    sm = load_model(args)
    pad = sm.tok.pad_token_id if sm.tok.pad_token_id is not None else 0
    bl = list(batches(docs, args.max_tokens))
    if args.limit_batches:
        bl = bl[:args.limit_batches]
    t0 = time.time()
    if args.mode in ("stats", "bench"):
        col = StatsCollector(sm, states, len(docs), sm.device)
        ntok = 0
        for bi, idx in enumerate(bl):
            ids, mask, valid, group, doc = make_batch(docs, idx, pad, sm.device, args.exclude_template)
            col.run_batch(ids, mask, valid, group, doc)
            ntok += int(mask.sum())
            if bi % 5 == 0 or bi == len(bl) - 1:
                el = time.time() - t0
                print(f"batch {bi + 1}/{len(bl)} tokens {ntok} states {len(states)} {el:.0f}s "
                      f"{ntok * len(states) / el:.0f} state-tok/s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB"
                      if torch.cuda.is_available() else f"batch {bi + 1}/{len(bl)} {el:.0f}s", flush=True)
        if args.mode == "stats":
            col.save(args.out)
            json.dump({"states": states, "n_docs": len(docs), "doc_ids": [d["doc_id"] for d in docs],
                       "tokens": ntok, "seconds": time.time() - t0, "smoke": args.smoke,
                       "exclude_template": args.exclude_template, "model": args.model,
                       "revision": args.revision}, open(args.out + ".meta.json", "w"), indent=1)
    else:
        neurons = [tuple(x) for x in json.load(open(args.neurons))]
        offsets, off = [], 0
        for d in docs:
            offsets.append(off)
            off += len(d["ids"])
        col = TraceCollector(sm, states, neurons, off, args.out)
        for bi, idx in enumerate(bl):
            ids, mask, *_ = make_batch(docs, idx, pad, sm.device)
            col.run_batch(ids, mask, [(offsets[i], len(docs[i]["ids"])) for i in idx])
            if bi % 10 == 0:
                print(f"traces batch {bi + 1}/{len(bl)} {time.time() - t0:.0f}s", flush=True)
        col.close()
        json.dump({"states": states, "neurons": neurons, "doc_ids": [d["doc_id"] for d in docs], "offsets": offsets,
                   "lengths": [len(d["ids"]) for d in docs], "seconds": time.time() - t0},
                  open(os.path.join(args.out, "meta.json"), "w"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
