"""Automated neuron investigator: an LLM proposes hypotheses and test texts, the live model measures them, repeat.

    python -m pn.investigate --run runs/v2 --neurons runs/v2_work/pilot.json --concurrency 64 --rounds 4 \
        --max-cost 10 [--model deepseek/deepseek-v3.2] [--device cuda] [--adapters /workspace/adapters]
    python -m pn.investigate ... --fake-llm      # canned LLM replies, no API calls, real (or tiny) model

Per neuron (default 4 rounds):
  round 1      blinded evidence packet (pn.explain.build_packet, discovery windows, adapters renamed to letters)
               -> hypothesis JSON with nickname, one-liner, adapter effects and 8 test texts in 4 minimal pairs
  measure      every test text in base (+ up to two lettered states the hypothesis names), activation at the target
               (max for high, min for negative, max |act| over the text for low), scored against the neuron's base
               p99 / p01 (detail quantiles_base), like rule v2 in pn/rescore.py
  rounds 2..N  packet + latest hypothesis + all result tables so far -> keep / revise / give_up and 8 new tests
  verdict      the last round's test pairs (never shown to the model) decide: held_up (>= 3 of 4 pairs pass),
               partly (3-5), not_held (<= 2), or no_coherent (the model gave up)

Text scoring (per text; base state only):
  high      value at target >= p99
  negative  value at target <= p01
  low       stays inside the partner's threshold over the whole text (reply only for assistant_reply tests), and
            the pair's contrast goes the right way (partner's target value beyond this text's extreme)

Measurement: one in-process StateModel (pn/states.py); a single worker batches pending texts across neurons and
states (one right-padded forward per state per chunk, stopped after the deepest requested layer), with down_proj
pre-hooks that keep only the requested neuron columns. Checked against pn.measure.Measurer at startup. Position 0
(attention sink) is reported as 0.

Outputs (runs/<run>/):
  neurons/cards/<id>.json            appended version (kind "investigation") and tests (rule "investigation")
  investigations/<id>.json           all rounds: prompts' variable parts, raw outputs, measurements, usage, cost
  investigations/finished.jsonl      one line per finished neuron (format fixed; the explorer reads it)
  investigations/progress.json       rewritten atomically every few seconds (format fixed; the explorer reads it)
Usage of every billed call goes to logs/api_usage.jsonl with component "pn.investigate".

Resumable: neurons with a finished investigation file are skipped. --max-cost caps the run's total spend (finished
neurons from earlier invocations included): no new neuron is launched once spend + running estimates + one more
neuron would exceed it; running neurons finish (hard stop for API calls at 1.2x the cap).
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pn import explain as ex

COMPONENT = "pn.investigate"
PROMPT_VERSION = "investigate-v1"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
AUTHOR = "{model} (investigator)"
N_TESTS, N_PAIRS = 8, 4
MAX_TOKENS = 1800
HARD_CEILING = 1.2           # API calls stop when spend reaches this multiple of --max-cost
SINK = "<|endoftext|>"
CHAT_PREFIX = "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
STATUS_TO_CARD = {"held_up": "tested", "partly": "mixed", "not_held": "rejected", "no_coherent": "untested"}
STATUS_KEYS = ("held_up", "partly", "not_held", "no_coherent")


def verdict(passes, n):
    """Final status from the final round's test pairs: a pair passes only if every text in it passes (the high or
    negative text reaches the neuron's base p99/p01 and the matched low text stays below it). held_up >= 3 of 4 pairs,
    partly 2 of 4 (thresholds scaled if fewer pairs survived)."""
    if n <= 0:
        return "no_coherent"
    if passes >= math.ceil(3 * n / 4):
        return "held_up"
    if passes >= math.ceil(2 * n / 4):
        return "partly"
    return "not_held"


# ----------------------------------------------------------------------------------------------------------------
# Rendering and scoring
# ----------------------------------------------------------------------------------------------------------------

def render(item):
    """Test item -> (full text fed to the model, char offset where the scored region starts)."""
    if item.get("context") == "assistant_reply":
        prefix = CHAT_PREFIX.format(u=item.get("user") or "")
        return prefix + item["text"], len(prefix)
    # a document-separator token takes the attention-sink position, so the first real word is measured normally
    return SINK + item["text"], len(SINK)


def region_start(tokens, char_offset):
    """Index of the first token that ends after char_offset in the joined token strings."""
    if char_offset <= 0:
        return 0
    pos = 0
    for k, t in enumerate(tokens):
        pos += len(t)
        if pos > char_offset:
            return k
    return len(tokens)


def thresholds(detail):
    """Base p99 / p01 of the neuron: quantiles_base (corpus), else natural_range, else the stored windows."""
    q = detail.get("quantiles_base")
    if q and "p99" in q and "p01" in q:
        return {"p99": float(q["p99"]), "p01": float(q["p01"]), "source": "quantiles_base"}
    q = detail.get("natural_range")
    if q and "p99" in q and "p01" in q:
        return {"p99": float(q["p99"]), "p01": float(q["p01"]), "source": "natural_range"}
    vals = sorted(v for e in detail.get("examples", []) for v in ((e.get("acts") or {}).get("base") or []))
    if not vals:
        raise ValueError(f"{detail.get('id')}: no quantiles and no stored activations")
    return {"p99": vals[min(len(vals) - 1, int(0.99 * len(vals)))], "p01": vals[int(0.01 * len(vals))],
            "source": "stored windows"}


def measure_value(m, item):
    """Activation of one measured text for its expectation: (value, idx_of_value, measured_at, region stats)."""
    rs = m.get("region_start", 0)
    toks, acts = m["tokens"][rs:], m["acts"][rs:]
    if not acts:
        return {"value": 0.0, "measured_at": "empty", "max": 0.0, "min": 0.0, "peak_token": "", "peak": 0.0}
    mx, mn = max(acts), min(acts)
    k_peak = max(range(len(acts)), key=lambda i: abs(acts[i]))
    idx = ex._target_tokens(toks, item["text"], item.get("target", "")) if item["expected"] != "low" else None
    region = idx or list(range(len(acts)))
    if item["expected"] == "high":
        val = max(acts[i] for i in region)
    elif item["expected"] == "negative":
        val = min(acts[i] for i in region)
    else:
        val = max(abs(mx), abs(mn))
    at = "target" if idx else ("text" if item["expected"] == "low" else "text (target not found)")
    return {"value": round(val, 4), "measured_at": at, "max": round(mx, 4), "min": round(mn, 4),
            "peak_token": toks[k_peak], "peak": round(acts[k_peak], 4)}


def score_texts(tests, base_vals, thr):
    """Rule-v2-style pass/fail per text. tests and base_vals are parallel lists. Returns list of (pass, threshold)."""
    p99, p01 = thr["p99"], thr["p01"]
    by_pair = defaultdict(list)
    for i, t in enumerate(tests):
        by_pair[t["pair"]].append(i)
    out = []
    for i, (t, v) in enumerate(zip(tests, base_vals)):
        partners = [j for j in by_pair[t["pair"]] if j != i]
        if t["expected"] == "high":
            out.append((v["value"] >= p99, f">= {p99:+.2f}"))
        elif t["expected"] == "negative":
            out.append((v["value"] <= p01, f"<= {p01:+.2f}"))
        else:
            hi = [j for j in partners if tests[j]["expected"] == "high"]
            ng = [j for j in partners if tests[j]["expected"] == "negative"]
            ok, conds = True, []
            if hi or not ng:
                ok &= v["max"] < p99
                ok &= all(base_vals[j]["value"] > v["max"] for j in hi)
                conds.append(f"max < {p99:+.2f}" + (" and < partner" if hi else ""))
            if ng or not hi:
                ok &= v["min"] > p01
                ok &= all(base_vals[j]["value"] < v["min"] for j in ng)
                conds.append(f"min > {p01:+.2f}" + (" and > partner" if ng else ""))
            out.append((bool(ok), " and ".join(conds)))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Measurement backends: async measure(layer, n, state, items) -> [{"tokens", "acts", "region_start"}]
# ----------------------------------------------------------------------------------------------------------------

class _StopForward(Exception):
    pass


class BatchedMeasurer:
    """Serves measurement requests from all neurons with one worker that batches across neurons and states."""

    def __init__(self, sm, max_batch_tokens=32768, gather_s=0.05, max_pending_texts=4096):
        self.sm = sm
        self.max_batch_tokens, self.gather_s, self.max_pending = max_batch_tokens, gather_s, max_pending_texts
        self.queue = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pn-gpu")
        self._decode_cache = {}
        self.pad_id = sm.tok.pad_token_id if sm.tok.pad_token_id is not None else (sm.tok.eos_token_id or 0)
        self.stats = {"batches": 0, "forwards": 0, "texts": 0, "unique_texts": 0, "padded_tokens": 0,
                      "real_tokens": 0, "gpu_s": 0.0}
        self._task = None

    def _decode(self, i):
        s = self._decode_cache.get(i)
        if s is None:
            s = self._decode_cache[i] = self.sm.tok.decode([i])
        return s

    def encode(self, item):
        full, off = render(item)
        ids = list(self.sm.tok(full).input_ids)
        toks = [self._decode(i) for i in ids]
        return ids, toks, region_start(toks, off)

    def start(self):
        self.queue = asyncio.Queue()
        self._task = asyncio.get_running_loop().create_task(self._worker())

    async def measure(self, layer, n, state, items):
        loop = asyncio.get_running_loop()
        futs = []
        for it in items:
            ids, toks, rs = self.encode(it)
            fut = loop.create_future()
            await self.queue.put((state, layer, n, ids, fut))
            futs.append((fut, toks, rs))
        out = []
        for fut, toks, rs in futs:
            acts = await fut
            out.append({"tokens": toks, "acts": acts, "region_start": rs})
        return out

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            t_end = loop.time() + self.gather_s
            while len(batch) < self.max_pending:
                try:
                    batch.append(self.queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                rem = t_end - loop.time()
                if rem <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), rem))
                except asyncio.TimeoutError:
                    break
            try:
                results = await loop.run_in_executor(self.executor, self._process, batch)
                for (_, _, _, _, fut), r in zip(batch, results):
                    if not fut.done():
                        fut.set_result(r)
            except Exception as e:  # deliver the error to every waiting neuron
                for *_, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _process(self, batch):
        """batch: [(state, layer, n, ids, fut)] -> list of activation lists (same order)."""
        import torch
        t0 = time.time()
        self.stats["batches"] += 1
        self.stats["texts"] += len(batch)
        results = [None] * len(batch)
        by_state = defaultdict(list)
        for k, (state, layer, n, ids, _) in enumerate(batch):
            by_state[state].append(k)
        with torch.inference_mode():
            for state, ks in by_state.items():
                self.sm.set_state(state)
                seqs = {}                                   # ids tuple -> {(layer, n)}
                for k in ks:
                    _, layer, n, ids, _ = batch[k]
                    seqs.setdefault(tuple(ids), set()).add((layer, n))
                self.stats["unique_texts"] += len(seqs)
                order = sorted(seqs, key=len)
                chunk = []
                for s in order + [None]:
                    if s is not None and (not chunk or (len(chunk) + 1) * len(s) <= self.max_batch_tokens):
                        chunk.append(s)
                        continue
                    got = self._forward(chunk, {c: seqs[c] for c in chunk})
                    for k in ks:
                        _, layer, n, ids, _ = batch[k]
                        key = tuple(ids)
                        if key in got:
                            results[k] = got[key][(layer, n)]
                    chunk = [s] if s is not None else []
        self.stats["gpu_s"] += time.time() - t0
        return results

    def _forward(self, seqs, wanted):
        """One right-padded forward over seqs; returns {seq: {(layer, n): acts list}} with position 0 set to 0."""
        import torch
        sm = self.sm
        cols = defaultdict(set)
        for w in wanted.values():
            for layer, n in w:
                cols[layer].add(n)
        cols = {l: sorted(v) for l, v in cols.items()}
        last = max(cols)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        cap, hooks = {}, []

        def make_hook(layer, idx):
            idx_t = torch.tensor(idx, device=sm.device)

            def hook(mod, inp):
                cap[layer] = inp[0].index_select(-1, idx_t).float().cpu()
                if layer == last:
                    raise _StopForward
            return hook
        for layer, idx in cols.items():
            hooks.append(sm.down_proj_module(layer).register_forward_pre_hook(make_hook(layer, idx)))
        try:
            sm.lm(input_ids=ids.to(sm.device), use_cache=False)
        except _StopForward:
            pass
        finally:
            for h in hooks:
                h.remove()
        self.stats["forwards"] += 1
        self.stats["padded_tokens"] += len(seqs) * T
        self.stats["real_tokens"] += sum(len(s) for s in seqs)
        out = {}
        for i, s in enumerate(seqs):
            d = {}
            for layer, n in wanted[s]:
                a = cap[layer][i, :len(s), cols[layer].index(n)].tolist()
                a[0] = 0.0
                d[(layer, n)] = a
            out[s] = d
        return out


class FuncMeasurer:
    """Wraps a pn.explain-style measure_fn(texts, neuron, state) -> [{"tokens", "acts"}] (tests, fakes)."""

    def __init__(self, fn):
        self.fn = fn
        self.stats = {"texts": 0}

    def start(self):
        pass

    async def measure(self, layer, n, state, items):
        out = []
        for it in items:
            full, off = render(it)
            m = self.fn([full], f"L{layer}/N{n}", state)[0]
            acts = list(m["acts"])
            if acts:
                acts[0] = 0.0
            out.append({"tokens": m["tokens"], "acts": acts, "region_start": region_start(m["tokens"], off)})
            self.stats["texts"] += 1
        await asyncio.sleep(0)
        return out


def validate_batched(bm, neurons, states, texts=None, tol=0.05):
    """Compare BatchedMeasurer with pn.measure.Measurer.acts on the same StateModel. Returns a report dict;
    report["ok"] is False if any token differs by more than tol x max(|ref| over the text, 1e-3 + 0.1)."""
    from pn.measure import Measurer
    texts = texts or [
        {"context": "plain", "text": "The darkness was so deep that nobody could see the path."},
        {"context": "plain", "text": "It's fine. I'm sure you'll do well tomorrow."},
        {"context": "plain", "text": "def mean(xs):\n    return sum(xs) / len(xs)\n"},
        {"context": "assistant_reply", "user": "My dog died yesterday.", "text": "Oh no, I'm so sorry. That is really hard."},
    ]
    ref_m = Measurer.__new__(Measurer)
    ref_m.sm, ref_m.calls = bm.sm, 0
    rows, ok = [], True
    for state in states:
        for layer, n in neurons:
            full = [render(t)[0] for t in texts]
            ref = ref_m.acts(full, f"L{layer}/N{n}", state)
            ids = [bm.encode(t)[0] for t in texts]
            got = bm._process([(state, layer, n, i, None) for i in ids])
            for t, r, g in zip(texts, ref, got):
                if len(r["acts"]) != len(g):
                    ok = False
                    rows.append({"state": state, "neuron": f"L{layer}/N{n}", "error": "length mismatch"})
                    continue
                scale = max(max(abs(x) for x in r["acts"]), 0.1)
                diff = max(abs(a - b) for a, b in zip(r["acts"], g))
                good = diff <= tol * scale
                ok &= good
                rows.append({"state": state, "neuron": f"L{layer}/N{n}", "text": t["text"][:30],
                             "max_abs_diff": round(diff, 5), "scale": round(scale, 4), "ok": good})
    return {"ok": bool(ok), "tol": tol, "rows": rows}


# ----------------------------------------------------------------------------------------------------------------
# LLM clients: async chat(messages, meta) -> {"content", "usage", "provider", "generation_id", "finish_reason"}
# ----------------------------------------------------------------------------------------------------------------

class LLMError(RuntimeError):
    pass


class OpenRouterLLM:
    def __init__(self, model, key, max_tokens=MAX_TOKENS, temperature=0.0, timeout=180.0, retries=5,
                 providers=None, json_mode=True, max_connections=128):
        import httpx
        self.model, self.key, self.max_tokens, self.temperature = model, key, max_tokens, temperature
        self.timeout, self.retries, self.providers, self.json_mode = timeout, retries, providers, json_mode
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "https://github.com/persona-neurons",
                     "X-Title": "persona-neurons investigate"},
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections))
        self._httpx = httpx

    def body(self, messages):
        b = {"model": self.model, "messages": messages, "temperature": self.temperature,
             "max_tokens": self.max_tokens, "usage": {"include": True},
             "reasoning": {"enabled": False, "exclude": True}}
        if self.json_mode:
            b["response_format"] = {"type": "json_object"}
        if self.providers:
            b["provider"] = {"order": list(self.providers), "allow_fallbacks": False}
        return b

    async def chat(self, messages, meta=None):
        last = None
        for attempt in range(self.retries):
            if attempt:
                await asyncio.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
            t0 = time.time()
            try:
                # wall-clock deadline: keep-alive bytes from OpenRouter can defeat the socket read timeout
                resp = await asyncio.wait_for(self.client.post(ex.OPENROUTER_URL, json=self.body(messages)),
                                              self.timeout)
            except (asyncio.TimeoutError, self._httpx.TransportError, self._httpx.TimeoutException) as e:
                last = f"transport: {type(e).__name__}: {e}"
                continue
            if resp.status_code in (400, 404) and self.json_mode and \
                    ("response_format" in resp.text or "No endpoints" in resp.text):
                self.json_mode = False   # pinned providers without JSON mode: rely on the prompt + parser
                last = f"HTTP {resp.status_code}; retrying without response_format"
                print(f"[investigate] {last}: {resp.text[:200]}", file=sys.stderr)
                continue
            if resp.status_code in (408, 429) or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if "error" in data and not data.get("choices"):
                last = f"API error: {json.dumps(data['error'])[:300]}"
                continue
            ch = (data.get("choices") or [{}])[0]
            return {"content": (ch.get("message") or {}).get("content"), "usage": ex._usage_of(data, self.model),
                    "provider": data.get("provider"), "generation_id": data.get("id"),
                    "finish_reason": ch.get("finish_reason"), "latency_s": round(time.time() - t0, 2)}
        raise LLMError(f"OpenRouter failed after {self.retries} attempts: {last}")

    async def close(self):
        await self.client.aclose()


class FakeLLM:
    """Canned, deterministic replies built from the packet text, for offline runs. Some neurons give up in round 2
    (int hash % 5 == 0) and some return an invalid first answer in round 1 (% 7 == 0), to exercise those paths."""

    def __init__(self, model="fake/canned", latency=(0.01, 0.05), cost_per_call=0.0005, give_up=None,
                 invalid_first=None):
        self.model, self.latency, self.cost_per_call = model, latency, cost_per_call
        self.give_up, self.invalid_first = give_up, invalid_first   # sets of neuron ids; None = hash rule
        self.calls = 0

    @staticmethod
    def _h(s):
        return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

    async def chat(self, messages, meta=None):
        meta = meta or {}
        await asyncio.sleep(random.uniform(*self.latency))
        self.calls += 1
        user = messages[1]["content"]
        rnd, nid, attempt = meta.get("round", 1), meta.get("neuron", ""), meta.get("attempt", 1)
        h = self._h(nid)
        toks = [json.loads(m) for m in re.findall(r'token=("(?:[^"\\]|\\.)*")', user)]
        words = [t.strip() for t in toks if re.fullmatch(r"[A-Za-z]{2,}", t.strip())] or ["word"]
        states = re.findall(r"state ([A-Z][+-]\d+(?:\.\d+)?)", user.split("\n## ")[0])
        invalid = (nid in self.invalid_first) if self.invalid_first is not None else h % 7 == 0
        gives_up = (nid in self.give_up) if self.give_up is not None else h % 5 == 0
        if rnd == 1 and invalid and attempt == 1:
            obj = {"label": "incomplete"}
        elif rnd == 2 and gives_up:
            obj = self._hyp(words, states, rnd, "give_up")
            obj["tests"], obj["no_coherent_hypothesis"] = [], True
        else:
            obj = self._hyp(words, states, rnd, "new" if rnd == 1 else ("keep" if rnd % 2 else "revise"))
        content = json.dumps(obj)
        inp = int(sum(len(m["content"]) for m in messages) / 3.6)
        out = int(len(content) / 3.6)
        return {"content": content, "provider": "fake", "generation_id": f"fake-{self.calls}", "finish_reason": "stop",
                "latency_s": 0.0, "usage": {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": 0,
                                            "cached_tokens": 0, "cost_usd": self.cost_per_call, "cost_source": "fake"}}

    def _hyp(self, words, states, rnd, decision):
        w = words[rnd % len(words)]
        tests = []
        for p in range(N_PAIRS):
            w = words[(rnd + p) % len(words)]
            if p % 2 == 0:
                tests += [{"pair": p + 1, "context": "plain", "user": "", "text": f"The note said {w} twice.",
                           "expected": "high", "target": w},
                          {"pair": p + 1, "context": "plain", "user": "", "text": "The weather was mild today.",
                           "expected": "low", "target": ""}]
            else:
                tests += [{"pair": p + 1, "context": "assistant_reply", "user": "Tell me something.",
                           "text": f"Sure, {w} is the word.", "expected": "high", "target": w},
                          {"pair": p + 1, "context": "assistant_reply", "user": "Tell me something.",
                           "text": "Sure, here is a plain answer.", "expected": "low", "target": ""}]
        return {"decision": decision, "label": f"the word {w!r}", "trigger_hypothesis": f"Fires on {w!r}.",
                "output_hypothesis": "Unknown.", "nickname": f'the "{w}" neuron',
                "one_liner": f"Lights up on the word \"{w}\".", "alternatives": ["topic", "position"],
                "adapter_changes": f"{states[0] if states else 'none'} makes it stronger.",
                "state_effects": [{"state": s, "direction": "stronger", "where": "text"} for s in states[:2]],
                "test_states": states[:2], "supporting": ["W01"], "contradicting": [], "confidence": "low",
                "no_coherent_hypothesis": False, "tests": tests}

    async def close(self):
        pass


# ----------------------------------------------------------------------------------------------------------------
# Prompts and validation
# ----------------------------------------------------------------------------------------------------------------

_READING = ex.SYSTEM_PROMPT.split("\nRules:")[0]
assert "How to read the evidence" in _READING

SYSTEM_PROMPT = _READING + """

You are investigating this neuron over several rounds. In each round you state a hypothesis and propose test texts.
The texts are run through the real model and the neuron's activation is measured. You then see the results and
revise. Thresholds come from the neuron's own activation distribution on ordinary text (given in the task).

Rules:
- Base neuron first. Describe both signs if both occur (what drives strong positive and strong negative values).
- Be brief: label at most 8 words; trigger_hypothesis and output_hypothesis at most two short sentences each;
  state_effects only for states with a clear change.
- nickname: 2 to 6 plain words, like 'the "so ___" neuron' or 'reply-opening "Oh"'. one_liner: one plain sentence
  a semi-technical reader understands, e.g. 'Fires on the word after a degree "so" ("so much", "so difficult"), not
  after "very".' Plain, matter-of-fact wording; no hype, no hedging filler.
- state_effects: for each lettered state you can say something about, how it changes the neuron: direction
  ("stronger", "weaker", "sign flip", "new trigger", "about the same") and where (e.g. "same tokens", "start of
  assistant replies", "emotional text"). adapter_changes: one or two sentences summarizing them.
- test_states: zero to two lettered states (written exactly as in the packet, e.g. "B+1") whose activations you want
  measured on your tests, in addition to base. Only base decides pass or fail; the other states are for you.
- tests: exactly 8 texts in 4 minimal pairs (pair 1-4, two texts each). The two texts of a pair differ minimally
  (one word or phrase) and have different expectations; at least one of them is "high" or "negative". expected is
  "high" (target reaches the positive threshold), "negative" (reaches the negative threshold) or "low" (stays
  within both thresholds over the whole text). target: the exact substring of the text where you expect the peak
  ("" for low). Texts at most 25 words.
- context: "plain" for ordinary text, or "assistant_reply" when the hypothesis is about chat: then "text" is the
  assistant's reply and "user" is a short user message; the model sees its chat template around them, and only the
  reply is scored.
- Choose tests that would fail if your hypothesis were wrong (word vs meaning, position, register, chat vs plain).
  Tests that are easy to pass teach nothing.
- If no consistent pattern exists, say so: decision "give_up" (from round 2 on) is an acceptable, useful answer.
- Output one minified JSON object and nothing else, with exactly these keys:
{"decision":"new|keep|revise|give_up","label":"...","trigger_hypothesis":"...","output_hypothesis":"...","nickname":"...","one_liner":"...","alternatives":["...","..."],"adapter_changes":"...","state_effects":[{"state":"B+1","direction":"stronger","where":"..."}],"test_states":["B+1"],"supporting":["W01"],"contradicting":["W02"],"confidence":"low|medium|high","no_coherent_hypothesis":false,"tests":[{"pair":1,"context":"plain","user":"","text":"...","expected":"high","target":"..."}]}"""

HYP_KEYS = ("label", "trigger_hypothesis", "output_hypothesis", "nickname", "one_liner", "alternatives",
            "adapter_changes", "state_effects", "test_states", "confidence", "no_coherent_hypothesis")
STR_KEYS = ("label", "trigger_hypothesis", "output_hypothesis", "nickname", "one_liner", "adapter_changes")
EXPECTED = ("high", "negative", "low")


def packet_body(packet):
    t = packet["text"]
    return t[:t.rfind("Return the JSON object now.")].rstrip() if "Return the JSON object now." in t else t


def scoring_section(thr):
    return (f"## Scoring of your tests (base state)\n"
            f"- high: activation at the target >= {thr['p99']:+.3f} (the neuron's 99th percentile on ordinary text)\n"
            f"- negative: activation at the target <= {thr['p01']:+.3f} (its 1st percentile)\n"
            f"- low: stays below {thr['p99']:+.3f} (partner high) or above {thr['p01']:+.3f} (partner negative) on every "
            f"token of the text, and the pair's contrast goes the right way.\n"
            f"Activations are per token; a target spanning several tokens counts its strongest token.")


def compact_hypothesis(h):
    return json.dumps({k: h.get(k) for k in HYP_KEYS}, ensure_ascii=False, separators=(",", ":"))


def results_table(rnd, rows, test_states):
    head = f"Round {rnd} results (measured states: base" + "".join(f", {s}" for s in test_states) + "):"
    lines = [head]
    for r in rows:
        ctx = f'reply to "{r["user"]}": ' if r["context"] == "assistant_reply" else "plain: "
        tgt = f' @"{r["target"]}"' if r["expected"] != "low" else ""
        val = f'{r["base"]["value"]:+.2f}'
        if r["expected"] != "low":
            val += f' ({r["base"]["measured_at"]})'
        st = "".join(f' | {s}={r["states"][s]["value"]:+.2f}' for s in test_states if s in r.get("states", {}))
        peak = f'; peak {r["base"]["peak"]:+.2f} at {json.dumps(r["base"]["peak_token"], ensure_ascii=False)}'
        lines.append(f'{r["pair"]}{"ab"[r["k"]]} {r["expected"]}{tgt}: base={val} need {r["threshold"]} '
                     f'{"PASS" if r["pass"] else "FAIL"}{st}{peak} :: {ctx}{json.dumps(r["text"], ensure_ascii=False)}')
    n = sum(r["pass"] for r in rows)
    lines.append(f"Round {rnd}: {n} of {len(rows)} texts passed.")
    return "\n".join(lines)


def round_user(packet, thr, rnd, n_rounds, hyp=None, tables=()):
    parts = [packet_body(packet), scoring_section(thr)]
    if rnd == 1:
        task = (f"## Task (round 1 of {n_rounds})\nState your hypothesis and propose 8 tests. Use decision \"new\".")
        if n_rounds == 1:
            task += " This is the only round: your tests decide the verdict."
    else:
        parts.append("## Your current hypothesis\n" + compact_hypothesis(hyp))
        parts.append("## Test results so far\n" + "\n\n".join(tables))
        final = rnd == n_rounds
        task = (f"## Task (round {rnd} of {n_rounds})\nCompare the results with your hypothesis. decision: \"keep\" if "
                f"it held, \"revise\" if the results call for a change (give the revised hypothesis), \"give_up\" if no "
                f"coherent hypothesis fits the evidence and results (tests may then be empty). Propose 8 new tests "
                f"(do not repeat earlier texts).")
        if final:
            task += (" This is the final round: you will not see these results, and they decide whether your "
                     "hypothesis held up. Give your best final hypothesis and tests that fairly check it.")
    parts.append(task + "\nReturn the JSON object now.")
    return "\n\n".join(parts)


def normalize_state(s, allowed):
    s = str(s).strip()
    s = re.sub(r"^state\s+", "", s, flags=re.I)
    return s if s in allowed else None


def make_validator(rnd, allowed_states, packet_wids, lenient=False):
    """Validator for one round. Repairs cosmetic issues in place; returns a list of errors that need a retry.
    lenient=True accepts high/negative targets that are not substrings (scored over the whole text instead)."""
    def validate(o):
        errs = []
        if not isinstance(o, dict):
            return ["expected a JSON object"]
        dec = str(o.get("decision", "new" if rnd == 1 else "")).strip().lower().replace(" ", "_")
        dec = {"giveup": "give_up", "give-up": "give_up"}.get(dec, dec)
        if rnd == 1 and dec not in ("new", "keep", "revise"):
            errs.append('decision must be "new" in round 1')
        if rnd > 1 and dec not in ("keep", "revise", "give_up"):
            errs.append('decision must be "keep", "revise" or "give_up"')
        o["decision"] = "new" if rnd == 1 else dec
        for k in STR_KEYS:
            if not isinstance(o.get(k), str) or (not o[k].strip() and k in ("label", "trigger_hypothesis")):
                if dec == "give_up" and k != "label":
                    o[k] = o.get(k) if isinstance(o.get(k), str) else ""
                else:
                    errs.append(f"{k}: missing or not a string")
        alts = o.get("alternatives")
        o["alternatives"] = [str(a) for a in alts] if isinstance(alts, list) else []
        o["confidence"] = o.get("confidence") if o.get("confidence") in ("low", "medium", "high") else "low"
        o["no_coherent_hypothesis"] = bool(o.get("no_coherent_hypothesis", False)) or dec == "give_up"
        se = o.get("state_effects")
        o["state_effects"] = [x for x in se if isinstance(x, dict)] if isinstance(se, list) else []
        ts = o.get("test_states") if isinstance(o.get("test_states"), list) else []
        ts = [normalize_state(s, allowed_states) for s in ts]
        o["test_states"] = list(dict.fromkeys(s for s in ts if s))[:2]
        for k in ("supporting", "contradicting"):
            v = o.get(k) if isinstance(o.get(k), list) else []
            o[k] = [str(x).strip() for x in v if str(x).strip() in packet_wids]
        tests = o.get("tests")
        if dec == "give_up":
            o["tests"] = []
            return errs
        if not isinstance(tests, list) or len(tests) != N_TESTS:
            return errs + [f"tests: need exactly {N_TESTS} texts in {N_PAIRS} pairs"]
        clean = []
        for i, t in enumerate(tests):
            if not isinstance(t, dict):
                errs.append(f"tests[{i}]: not an object")
                continue
            ctx = t.get("context") if t.get("context") in ("plain", "assistant_reply") else "plain"
            text, exp = t.get("text"), str(t.get("expected", "")).strip().lower()
            tgt = t.get("target") if isinstance(t.get("target"), str) else ""
            user = t.get("user") if isinstance(t.get("user"), str) else ""
            try:
                pair = int(t.get("pair", i // 2 + 1))
            except (TypeError, ValueError):
                pair = i // 2 + 1
            if not isinstance(text, str) or not text.strip():
                errs.append(f"tests[{i}]: empty text")
                continue
            if len(text) > 600 or len(user) > 400:
                errs.append(f"tests[{i}]: text too long")
            if exp not in EXPECTED:
                errs.append(f"tests[{i}]: expected must be high, negative or low")
            if exp in ("high", "negative") and not lenient and \
                    (not tgt.strip() or tgt.strip().lower() not in text.lower()):
                errs.append(f"tests[{i}]: target {tgt!r} is not a substring of its text")
            if ctx == "assistant_reply" and not user.strip():
                errs.append(f"tests[{i}]: assistant_reply needs a user message")
            clean.append({"pair": pair, "context": ctx, "user": user if ctx == "assistant_reply" else "",
                          "text": text, "expected": exp, "target": tgt if exp != "low" else ""})
        if errs:
            return errs
        pairs = defaultdict(list)
        for t in clean:
            pairs[t["pair"]].append(t)
        if len(pairs) != N_PAIRS or any(len(v) != 2 for v in pairs.values()):
            # repair: pair consecutive texts if the model's pair numbers are inconsistent
            for i, t in enumerate(clean):
                t["pair"] = i // 2 + 1
            pairs = defaultdict(list)
            for t in clean:
                pairs[t["pair"]].append(t)
        bad = []
        for p, v in pairs.items():
            exps = [t["expected"] for t in v]
            if len(set(exps)) < 2 or all(e == "low" for e in exps):
                bad.append(p)
                errs.append(f"pair {p}: the two texts need different expectations, at least one high or negative")
        if lenient and bad and len(pairs) - len(bad) >= 2:
            # salvage: drop malformed pairs instead of failing the neuron; the verdict scales with the pairs kept
            clean = [t for t in clean if t["pair"] not in bad]
            errs = [e for e in errs if not e.startswith("pair ")]
            o["dropped_pairs"] = bad
        o["tests"] = sorted(clean, key=lambda t: t["pair"])
        return errs
    return validate


# ----------------------------------------------------------------------------------------------------------------
# Investigator
# ----------------------------------------------------------------------------------------------------------------

def nid_of(layer, n):
    return f"L{layer}/N{n}"


def load_neuron_list(path, index=None):
    """[[layer, n], ...] or [{layer, n, kind, trait}, ...] -> list of {id, layer, n, kind, trait} (order kept)."""
    index = index or {}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("neurons") or raw.get("items") or []
    out, seen = [], set()
    for r in raw:
        if isinstance(r, (list, tuple)):
            layer, n, kind, trait = int(r[0]), int(r[1]), None, None
        elif isinstance(r, str):
            m = re.match(r"L(\d+)/N(\d+)", r)
            layer, n, kind, trait = int(m.group(1)), int(m.group(2)), None, None
        else:
            layer, n, kind, trait = int(r["layer"]), int(r["n"]), r.get("kind"), r.get("trait")
        nid = nid_of(layer, n)
        if nid in seen:
            continue
        seen.add(nid)
        row = index.get(nid) or {}
        if kind is None:
            kind = "random" if "random" in str(row.get("kind", "")) else "trait"
            trait = (row.get("selected_for") or [None])[0]
        out.append({"id": nid, "layer": layer, "n": n, "kind": kind if kind in ("trait", "random") else "trait",
                    "trait": trait})
    return out


def _empty_counts():
    return {"done": 0, **{k: 0 for k in STATUS_KEYS}}


def decode_letters(text, amap):
    """'B+1 is stronger' -> 'n_amp+1 is stronger' using the packet's adapter map."""
    inv = {v: k for k, v in (amap or {}).items()}
    if not inv or not isinstance(text, str):
        return text
    return re.sub(r"\b(?:state )?([A-Z])([+-]\d+(?:\.\d+)?)\b",
                  lambda m: f"{inv[m.group(1)]}{m.group(2)}" if m.group(1) in inv else m.group(0), text)


class Investigator:
    def __init__(self, run_dir, llm, measurer, rounds=4, concurrency=64, max_cost=10.0, n_windows=30,
                 usage_log=None, progress_every=3.0, measure_label="live", log=print):
        self.run_dir = Path(run_dir)
        self.inv_dir = self.run_dir / "investigations"
        self.inv_dir.mkdir(parents=True, exist_ok=True)
        self.llm, self.measurer = llm, measurer
        self.rounds, self.concurrency, self.max_cost, self.n_windows = rounds, concurrency, max_cost, n_windows
        self.usage_log = Path(usage_log) if usage_log else ex.USAGE_LOG
        self.progress_every, self.measure_label, self.log = progress_every, measure_label, log
        self.model = getattr(llm, "model", "unknown")
        self.run_id = self.run_dir.name
        self.index = ex.load_index(self.run_dir)
        self.session_spent = 0.0
        self.prior_cost = 0.0
        self.running = {}           # id -> spent so far
        self.recs = {}              # id -> in-memory investigation record of running neurons
        self.finished_rows = {}     # id -> finished.jsonl row (prior + this session)
        self.n_failed = 0
        self.session_done = 0
        self.stop_launch = None
        self.started = ex.now()
        self.t0 = time.time()
        self.items = []
        self.prior_estimate = None

    # ---- files

    def inv_path(self, nid):
        return self.inv_dir / (ex.nid_file(nid) + ".json")

    def _load_finished(self):
        rows = {}
        fp = self.inv_dir / "finished.jsonl"
        if fp.exists():
            for line in fp.read_text().splitlines():
                try:
                    r = json.loads(line)
                    rows[r["id"]] = r
                except (ValueError, KeyError):
                    continue
        # reconcile: finished investigation files whose line was never appended (crash between the writes)
        for p in self.inv_dir.glob("L*_N*.json"):
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            if d.get("finished") and d["id"] not in rows and d.get("finished_row"):
                self._append_finished(d["finished_row"])
                rows[d["id"]] = d["finished_row"]
        return rows

    def _append_finished(self, row):
        with open(self.inv_dir / "finished.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    def _log_usage(self, rec):
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- cost

    def total_spent(self):
        return self.prior_cost + self.session_spent

    def est_per_neuron(self):
        costs = [r.get("cost_usd") or 0.0 for r in self.finished_rows.values() if (r.get("cost_usd") or 0) > 0]
        if len(costs) >= 3:
            return 1.25 * sum(costs) / len(costs)
        return self.prior_estimate

    def projected(self):
        est = self.est_per_neuron()
        return self.total_spent() + sum(max(0.0, est - s) for s in self.running.values()) + est

    # ---- progress

    def progress(self):
        by_kind = {"trait": _empty_counts(), "random": _empty_counts()}
        by_trait = {t: _empty_counts() for t in sorted({i["trait"] for i in self.items if i["trait"]})}
        ids = {i["id"] for i in self.items}
        done_costs = []
        for nid, r in self.finished_rows.items():
            if nid not in ids:
                continue
            st = r.get("status")
            k = by_kind.setdefault(r.get("kind") or "trait", _empty_counts())
            k["done"] += 1
            if st in STATUS_KEYS:
                k[st] += 1
            if r.get("trait"):
                t = by_trait.setdefault(r["trait"], _empty_counts())
                t["done"] += 1
                if st in STATUS_KEYS:
                    t[st] += 1
            done_costs.append(r.get("cost_usd") or 0.0)
        n_done = sum(v["done"] for v in by_kind.values())
        n_total = len(self.items)
        elapsed_min = (time.time() - self.t0) / 60
        eta = None
        if self.session_done > 0 and elapsed_min > 0:
            remaining = len(self.running) if self.stop_launch else max(0, n_total - n_done - self.n_failed)
            eta = round(remaining / (self.session_done / elapsed_min), 1)
        elif n_total and n_done + self.n_failed >= n_total:
            eta = 0.0
        est = (sum(done_costs) / len(done_costs)) if done_costs else self.prior_estimate
        return {"started": self.started, "updated": ex.now(), "n_total": n_total, "n_done": n_done,
                "n_running": len(self.running), "n_failed": self.n_failed, "cost_usd": round(self.total_spent(), 6),
                "cost_cap_usd": self.max_cost, "est_cost_per_neuron": None if est is None else round(est, 6),
                "eta_minutes": eta, "by_kind": by_kind, "by_trait": by_trait}

    def write_progress(self):
        ex._write_json(self.inv_dir / "progress.json", self.progress())

    async def _progress_loop(self):
        while True:
            try:
                self.write_progress()
            except Exception as e:  # never let progress writing kill the run
                print(f"[investigate] progress write failed: {e}", file=sys.stderr)
            await asyncio.sleep(self.progress_every)

    # ---- LLM call with validation and one retry

    async def call(self, rec, messages, validator, rnd, salvage=None):
        nid = rec["id"]
        attempts, parsed, errors = [], None, ["not called"]
        msgs = list(messages)
        for attempt in (1, 2):
            if self.total_spent() >= HARD_CEILING * self.max_cost:
                raise LLMError(f"hard cost ceiling reached ({self.total_spent():.4f} >= {HARD_CEILING} x cap)")
            r = await self.llm.chat(msgs, {"neuron": nid, "round": rnd, "attempt": attempt})
            u = r["usage"]
            self.session_spent += u["cost_usd"]
            self.running[nid] = self.running.get(nid, 0.0) + u["cost_usd"]
            for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "cost_usd"):
                rec["usage"][k] = rec["usage"].get(k, 0) + (u.get(k) or 0)
            content = r["content"]
            try:
                parsed = ex._parse_json(content)
                errors = validator(parsed)
            except Exception as e:
                parsed, errors = None, [f"parse error: {e}"]
            self._log_usage({"ts": ex.now(), "component": COMPONENT, "purpose": f"investigate:round{rnd}",
                             "run_id": self.run_id, "model": self.model, "provider": r.get("provider"),
                             "generation_id": r.get("generation_id"), "neuron": nid, "round": rnd,
                             "attempt": attempt, "finish_reason": r.get("finish_reason"), "valid": not errors,
                             "latency_s": r.get("latency_s"), **u})
            attempts.append({"attempt": attempt, "content": content, "errors": errors, "usage": u,
                             "provider": r.get("provider"), "generation_id": r.get("generation_id"),
                             "finish_reason": r.get("finish_reason")})
            if not errors:
                break
            if r.get("finish_reason") == "length" or not content:
                msgs = messages + [{"role": "user", "content": "Your previous answer was cut off by the length limit. "
                                    "Answer again as one minified JSON object, more briefly (short hypotheses, short "
                                    "texts)."}]
            else:
                msgs = messages + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": "That output was invalid: " + "; ".join(errors[:6])
                                    + ". Return only one complete minified JSON object with all keys."}]
        if errors and salvage is not None and isinstance(parsed, dict) and \
                all("not a substring" in e for e in errors) and not salvage(parsed):
            attempts[-1]["salvaged"] = errors   # targets not found: those texts are scored over the whole text
            errors = []
        return parsed, errors, attempts

    # ---- measurement for one round

    async def measure_round(self, item, tests, real_states, thr):
        layer, n = item["layer"], item["n"]
        states = ["base"] + [s for s in real_states if s != "base"]
        meas = await asyncio.gather(*(self.measurer.measure(layer, n, s, tests) for s in states))
        by_state = dict(zip(states, meas))
        base_vals = [measure_value(m, t) for m, t in zip(by_state["base"], tests)]
        scores = score_texts(tests, base_vals, thr)
        rows, seen_pair = [], defaultdict(int)
        for i, t in enumerate(tests):
            k = seen_pair[t["pair"]]
            seen_pair[t["pair"]] += 1
            st = {s: measure_value(by_state[s][i], t) for s in states if s != "base"}
            rows.append({**t, "k": min(k, 1), "base": base_vals[i], "threshold": scores[i][1],
                         "pass": bool(scores[i][0]), "states_real": st,
                         "n_tokens": len(by_state["base"][i]["tokens"])})
        return rows

    # ---- one neuron

    async def investigate(self, item):
        nid = item["id"]
        inv_id = uuid.uuid4().hex[:12]
        rec = {"id": nid, "investigation_id": inv_id, "kind": item["kind"], "trait": item["trait"],
               "model": self.model, "prompt_version": PROMPT_VERSION, "started": ex.now(), "finished": False,
               "rounds_planned": self.rounds, "rounds": [], "usage": {"cost_usd": 0.0}, "measure": self.measure_label}
        self.recs[nid] = rec
        p = self.inv_path(nid)
        if p.exists():   # spend of an earlier failed attempt stays in this neuron's cost and in the run total
            try:
                prev = float((json.loads(p.read_text()).get("usage") or {}).get("cost_usd") or 0.0)
            except (ValueError, TypeError):
                prev = 0.0
            if prev:
                rec["usage"].update(cost_usd=prev, cost_usd_earlier_attempts=prev)
                self.session_spent += prev
                self.running[nid] = self.running.get(nid, 0.0) + prev
        detail = ex.load_detail(self.run_dir, nid)
        packet = ex.build_packet(detail, "blind", self.n_windows, index_row=self.index.get(nid))
        thr = thresholds(detail)
        smap, amap = packet["state_map"], packet["adapter_map"]
        inv_smap = {v: k for k, v in smap.items()}
        allowed = sorted(v for v in smap.values() if v != "base")
        wids = {w["wid"] for w in packet["windows"]}
        rec.update(packet_hash=packet["hash"], packet_est_tokens=packet["est_tokens"], state_map=smap,
                   adapter_map=amap, thresholds=thr, n_windows=len(packet["windows"]))
        hyp, tables, status, final_rows = None, [], None, None
        for rnd in range(1, self.rounds + 1):
            user = round_user(packet, thr, rnd, self.rounds, hyp, tables)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
            parsed, errors, attempts = await self.call(rec, msgs, make_validator(rnd, allowed, wids), rnd,
                                                       salvage=make_validator(rnd, allowed, wids, lenient=True))
            rrec = {"round": rnd, "user_tail": user[len(packet_body(packet)):], "attempts": attempts,
                    "valid": not errors, "errors": errors,
                    "usage": {k: sum((a["usage"].get(k) or 0) for a in attempts)
                              for k in ("input_tokens", "output_tokens", "cached_tokens", "cost_usd")}}
            rec["rounds"].append(rrec)
            if errors:
                self._write_partial(rec)
                raise LLMError(f"round {rnd}: invalid output after retry: {errors[:3]}")
            rrec["hypothesis"] = {k: parsed.get(k) for k in HYP_KEYS + ("decision", "supporting", "contradicting")}
            if parsed["decision"] == "give_up":
                status = "no_coherent"
                hyp = parsed if hyp is None else {**hyp, **{k: v for k, v in parsed.items() if v}}
                hyp["no_coherent_hypothesis"] = True
                self._write_partial(rec)
                break
            hyp = parsed
            real_states = [inv_smap[s] for s in parsed["test_states"] if s in inv_smap]
            rows = await self.measure_round(item, parsed["tests"], real_states, thr)
            for r in rows:   # blind names for the table the model sees
                r["states"] = {smap.get(s, s): v for s, v in r["states_real"].items()}
            pairs = defaultdict(list)
            for t, r in zip(parsed["tests"], rows):
                pairs[t.get("pair")].append(bool(r["pass"]))
            rrec.update(tests=parsed["tests"], measured_states=["base"] + real_states, results=rows,
                        passes=sum(all(v) for v in pairs.values()), n_tests=len(pairs),
                        text_passes=sum(r["pass"] for r in rows), n_texts=len(rows))
            tables.append(results_table(rnd, rows, parsed["test_states"]))
            final_rows = rows
            self._write_partial(rec)
        if status is None:
            final = rec["rounds"][-1]
            status = verdict(final["passes"], final["n_tests"])
            passes, n_final = final["passes"], final["n_tests"]
        else:
            passes, n_final = None, 0
        rec["status"] = status
        rec["card_status"] = STATUS_TO_CARD[status]
        rec["passes"], rec["n_final_tests"] = passes, n_final
        rec["rounds_done"] = len(rec["rounds"])
        card_v = self._write_card(item, rec, hyp, packet)
        rec["card_version"] = card_v
        rec["finished_at"] = ex.now()
        row = {"id": nid, "kind": item["kind"], "trait": item["trait"], "nickname": (hyp or {}).get("nickname", ""),
               "one_liner": (hyp or {}).get("one_liner", ""), "status": status, "passes": passes,
               "n_final_tests": n_final, "rounds": rec["rounds_done"], "cost_usd": round(rec["usage"]["cost_usd"], 6),
               "finished_at": rec["finished_at"]}
        rec["finished"], rec["finished_row"] = True, row
        ex._write_json(self.inv_path(nid), rec)
        self._append_finished(row)
        return row

    def _write_partial(self, rec):
        ex._write_json(self.inv_path(rec["id"]), rec)

    def _write_card(self, item, rec, hyp, packet):
        nid = item["id"]
        amap = packet["adapter_map"]
        card = ex.load_card(self.run_dir, nid)
        v = max([x.get("v", 0) for x in card["versions"]] + [0]) + 1
        wmap = {w["wid"]: w["ex_id"] for w in packet["windows"]}
        h = hyp or {}
        final_tests = rec["rounds"][-1].get("tests") or []
        version = {
            "v": v, "author": AUTHOR.format(model=self.model), "created": ex.now(), "kind": "investigation",
            "blind": True, "label": h.get("label") or "no coherent hypothesis",
            "nickname": h.get("nickname", ""), "one_liner": h.get("one_liner", ""),
            "trigger_hypothesis": h.get("trigger_hypothesis", ""), "output_hypothesis": h.get("output_hypothesis", ""),
            "alternatives": (h.get("alternatives") or [])[:3],
            "adapter_changes": decode_letters(h.get("adapter_changes", ""), amap),
            "adapter_changes_blind": h.get("adapter_changes", ""),
            "state_effects": [{**e, "state": decode_letters(str(e.get("state", "")), amap)}
                              for e in h.get("state_effects") or []],
            "supporting": [wmap[w] for w in h.get("supporting") or [] if w in wmap],
            "contradicting": [wmap[w] for w in h.get("contradicting") or [] if w in wmap],
            "proposed_tests": [f"[{t['expected']}] {t['text']}" + (f" @{t['target']!r}" if t["target"] else "")
                               for t in final_tests],
            "confidence": h.get("confidence", "low"),
            "no_coherent_hypothesis": rec["status"] == "no_coherent",
            "investigation": {"id": rec["investigation_id"], "status": rec["status"], "passes": rec["passes"],
                              "n_final_tests": rec["n_final_tests"], "rounds": len(rec["rounds"]),
                              "file": f"investigations/{ex.nid_file(nid)}.json"},
            "usage": {k: rec["usage"].get(k, 0) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                                          "cost_usd")},
            "packet_hash": packet["hash"], "packet_mode": packet["mode"], "state_map": packet["state_map"],
            "prompt_version": PROMPT_VERSION,
        }
        card["versions"].append(version)
        n_rounds = len(rec["rounds"])
        for rr in rec["rounds"]:
            rows = rr.get("results") or []
            final = rr["round"] == n_rounds and rec["status"] != "no_coherent"
            label = (rr.get("hypothesis") or {}).get("label", "")
            for p in sorted({r["pair"] for r in rows}):
                prs = [r for r in rows if r["pair"] == p]
                npass = sum(r["pass"] for r in prs)
                card["tests"].append({
                    "test_id": f"inv{v}-r{rr['round']}-p{p}", "card_version": v, "rule": "investigation",
                    "round": rr["round"], "final": final,
                    "description": f"round {rr['round']} ({label})",
                    "prediction": "; ".join(f"{r['expected']} on {r['target'] or 'whole text'!r}" for r in prs),
                    "texts": [{"text": r["text"], "context": r["context"], "user": r["user"],
                               "expected": r["expected"], "target": r["target"], "value": r["base"]["value"],
                               "threshold": r["threshold"], "pass": r["pass"], "measured_max": r["base"]["max"],
                               "measured_min": r["base"]["min"], "measured_at": r["base"]["measured_at"],
                               "states": r["states_real"]} for r in prs],
                    "result": "pass" if npass == len(prs) else ("fail" if npass == 0 else "mixed"),
                    "state": "base", "thresholds": {"p99": rec["thresholds"]["p99"], "p01": rec["thresholds"]["p01"],
                                                    "source": rec["thresholds"]["source"]},
                    "notes": f"measure={self.measure_label}; investigation {rec['investigation_id']}",
                    "created": ex.now()})
        card["status"] = STATUS_TO_CARD[rec["status"]]
        card.setdefault("status_history", []).append({"rule": "investigation", "status": card["status"],
                                                      "pass": rec["passes"], "of": rec["n_final_tests"],
                                                      "version": v})
        ex.save_card(self.run_dir, card)
        return v

    # ---- driver

    async def _run_one(self, item, sem):
        nid = item["id"]
        self.running[nid] = 0.0
        try:
            row = await self.investigate(item)
            self.finished_rows[nid] = row
            self.session_done += 1
            self.log(f"[done] {nid:12s} {row['status']:12s} {row['passes']}/{row['n_final_tests']} "
                     f"rounds={row['rounds']} ${row['cost_usd']:.4f} {row['nickname']!r}")
        except Exception as e:
            self.n_failed += 1
            tb = traceback.format_exc()
            self.log(f"[fail] {nid}: {type(e).__name__}: {e}")
            try:
                p = self.inv_path(nid)
                d = self.recs.get(nid) or (json.loads(p.read_text()) if p.exists() else {"id": nid})
                if not d.get("finished"):
                    d.update(finished=False, error=f"{type(e).__name__}: {e}", traceback=tb, failed_at=ex.now())
                    ex._write_json(p, d)
            except Exception:
                pass
        finally:
            self.running.pop(nid, None)
            self.recs.pop(nid, None)
            sem.release()

    async def run(self, items, prior_estimate=None):
        self.items = items
        self.finished_rows = self._load_finished()
        ids = {i["id"] for i in items}
        self.prior_cost = sum((r.get("cost_usd") or 0.0) for k, r in self.finished_rows.items() if k in ids)
        todo = [i for i in items if i["id"] not in self.finished_rows]
        if prior_estimate is None:
            # upper bound until real costs exist: full packet + tables each round, max output, one retry in two
            est_in = 9000 + 1200 * self.rounds
            prior_estimate = 1.5 * self.rounds * ex.estimate_cost(self.model, est_in, 1100)
        self.prior_estimate = prior_estimate
        self.log(f"[start] {len(items)} neurons, {len(items) - len(todo)} already finished, {len(todo)} to do; "
                 f"cap ${self.max_cost:.2f}, spent so far ${self.prior_cost:.4f}, est/neuron ${prior_estimate:.4f}")
        self.measurer.start()
        prog = asyncio.get_running_loop().create_task(self._progress_loop())
        sem = asyncio.Semaphore(self.concurrency)
        tasks = []
        try:
            for item in todo:
                await sem.acquire()
                if self.projected() > self.max_cost:
                    self.stop_launch = f"cost cap: projected ${self.projected():.4f} > ${self.max_cost:.2f}"
                    self.log(f"[stop] not launching more neurons ({self.stop_launch})")
                    sem.release()
                    break
                tasks.append(asyncio.get_running_loop().create_task(self._run_one(item, sem)))
            await asyncio.gather(*tasks)
        finally:
            prog.cancel()
            self.write_progress()
        return self.progress()


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def build_state_model(base, adapters, revision, device):
    from pn.collect import ADAPTER_DIRS
    from pn.states import StateModel
    ad = {k: os.path.join(adapters, k if os.path.isdir(os.path.join(adapters, k)) else v)
          for k, v in ADAPTER_DIRS.items()}
    missing = [p for p in ad.values() if not os.path.isdir(p)]
    if missing:
        raise SystemExit(f"adapter directories not found: {missing}")
    rev = None if os.path.isdir(base) else revision
    return StateModel(base, ad, revision=rev, device=device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--neurons", required=True, help="JSON list of [layer, n] or {layer, n, kind, trait}")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=10.0, help="USD cap for the run's total spend")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--providers", default=None, help="comma list of OpenRouter providers to pin ('' for none); "
                    "default: pn.judge.DEFAULT_PROVIDERS for the model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="base model id or local dir")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--adapters", default="/workspace/adapters")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=180.0, help="hard per-request wall-clock timeout (s)")
    ap.add_argument("--max-batch-tokens", type=int, default=32768, help="padded tokens per forward pass")
    ap.add_argument("--n-windows", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="only the first N neurons of the list")
    ap.add_argument("--fake-llm", action="store_true", help="canned LLM replies (no API calls, no spend)")
    ap.add_argument("--no-validate", action="store_true", help="skip the batched-vs-Measurer check at startup")
    ap.add_argument("--validate-tol", type=float, default=0.05)
    ap.add_argument("--usage-log", default=None)
    ap.add_argument("--progress-every", type=float, default=3.0)
    a = ap.parse_args(argv)

    run_dir = ex.resolve_run(a.run)
    index = ex.load_index(run_dir)
    items = load_neuron_list(a.neurons, index)
    if a.limit:
        items = items[:a.limit]
    missing = [i["id"] for i in items if not (run_dir / "neurons" / "detail" / (ex.nid_file(i["id"]) + ".json")).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} neurons have no detail file in {run_dir}: {missing[:5]}")

    sm = build_state_model(a.base, a.adapters, a.revision, a.device)
    bm = BatchedMeasurer(sm, max_batch_tokens=a.max_batch_tokens)
    if not a.no_validate:
        nv = [(i["layer"], i["n"]) for i in items[:3]]
        rep = validate_batched(bm, nv, ["base", "n_amp+1"], tol=a.validate_tol)
        worst = max((r.get("max_abs_diff", float("inf")) / r.get("scale", 1)) for r in rep["rows"])
        print(f"[validate] batched vs pn.measure.Measurer: ok={rep['ok']}, worst diff {worst:.4f} x scale "
              f"over {len(rep['rows'])} (text, neuron, state)", flush=True)
        ex._write_json(run_dir / "investigations" / "validation.json", {"created": ex.now(), **rep})
        if not rep["ok"]:
            raise SystemExit("batched measurement does not match Measurer; see investigations/validation.json")

    if a.fake_llm:
        llm = FakeLLM()
        usage_log = a.usage_log or (run_dir / "investigations" / "fake_usage.jsonl")
    else:
        from pn.judge import DEFAULT_PROVIDERS
        prov = DEFAULT_PROVIDERS.get(a.model) if a.providers is None else [p for p in a.providers.split(",") if p]
        llm = OpenRouterLLM(a.model, ex._api_key(), max_tokens=a.max_tokens, temperature=a.temperature,
                            timeout=a.timeout, providers=prov, max_connections=max(16, 2 * a.concurrency))
        usage_log = a.usage_log
    label = f"live {os.path.basename(os.path.normpath(a.base))}" + (f"@{a.revision[:7]}" if not os.path.isdir(a.base) else "")
    inv = Investigator(run_dir, llm, bm, rounds=a.rounds, concurrency=a.concurrency, max_cost=a.max_cost,
                       n_windows=a.n_windows, usage_log=usage_log, progress_every=a.progress_every,
                       measure_label=label, log=lambda s: print(s, flush=True))

    async def go():
        try:
            return await inv.run(items)
        finally:
            await llm.close()
    prog = asyncio.run(go())
    print(json.dumps({k: prog[k] for k in ("n_total", "n_done", "n_failed", "cost_usd", "est_cost_per_neuron")}))
    print(f"[gpu] {json.dumps(bm.stats)}")


if __name__ == "__main__":
    main()
