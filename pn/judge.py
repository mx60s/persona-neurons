"""Blinded LLM judge for the behavioral evaluation (OpenRouter).

    python -m pn.judge --run runs/v1                       # reads runs/v1/behavior/{prompts.json,responses.jsonl}
    python -m pn.judge --run runs/v1 --model qwen/qwen3-235b-a22b-2507 --concurrency 16 --max-cost 2.0

Each request scores one response for one trait (1-7) plus coherence (1-5) with the rubric in data/rubrics.md and
returns JSON. A response is scored on its prompt's trait, and on both traits when the prompt is in the common subset.

Blinding: the judge sees only the user question and the response text. Every distinct (question, response) pair gets
a blind id derived from a per-run random salt, the request order is shuffled, and nothing about the state,
intervention or prompt id is sent.

Caching: results are cached by (content hash, trait, judge model, rubric hash) in behavior/judge_cache.jsonl, so a
rerun (or identical text produced by two states) is never billed twice. Every billed request appends a line to
logs/api_usage.jsonl with OpenRouter's usage and cost fields. Requests that still fail after retries are written to
behavior/judge_failures.jsonl and left out of scores.jsonl.

Outputs (docs/run_format.md): behavior/scores.jsonl and behavior/summary.json (per trait and state: mean score and
coherence with bootstrap 90% CIs over prompts, plus paired differences from base).
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import secrets
import sys
import time
from datetime import datetime, timezone

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRICS = os.path.join(REPO, "data", "rubrics.md")
USAGE_LOG = os.path.join(REPO, "logs", "api_usage.jsonl")
KEYS_ENV = os.path.expanduser("~/.config/persona-neurons/keys.env")
URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
# Provider pin for the default model: GMICloud (OpenRouter's cheapest route) ignores reasoning.enabled=false and still
# emits reasoning tokens; these honour it (checked 2026-10-25). fp8 first so the weights match across fallbacks.
DEFAULT_PROVIDERS = {DEFAULT_MODEL: ["atlas-cloud/fp8", "siliconflow/fp8", "baidu/fp8"]}
TRAITS = ("openness", "neuroticism")


# ---------------------------------------------------------------- rubric, key, hashing

def load_rubrics(path=RUBRICS):
    """{trait: prompt template} from the ```judge-prompt <trait> blocks, plus a hash of all of them."""
    text = open(path, encoding="utf-8").read()
    blocks = dict(re.findall(r"```judge-prompt (\w+)\n(.*?)\n```", text, flags=re.S))
    for t in TRAITS:
        if t not in blocks or "<<QUESTION>>" not in blocks[t] or "<<RESPONSE>>" not in blocks[t]:
            raise ValueError(f"rubric for {t} missing or lacks placeholders in {path}")
    h = hashlib.sha256(json.dumps(blocks, sort_keys=True).encode()).hexdigest()[:12]
    return blocks, h


def render(template, question, response):
    return template.replace("<<QUESTION>>", question).replace("<<RESPONSE>>", response)


def load_key():
    """OPENROUTER_API_KEY from keys.env (preferred: the shell may hold a stale key), else the environment."""
    key = None
    if os.path.exists(KEYS_ENV):
        for line in open(KEYS_ENV):
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    key = key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(f"OPENROUTER_API_KEY not set and not found in {KEYS_ENV}")
    return key


def content_hash(question, response):
    return hashlib.sha256((question + "\x00" + response).encode("utf-8")).hexdigest()


def cache_key(chash, trait, judge, rubric_hash):
    return hashlib.sha256(f"{chash}|{trait}|{judge}|{rubric_hash}".encode()).hexdigest()


def blind_id(salt, chash):
    return "b" + hashlib.sha256((salt + chash).encode()).hexdigest()[:10]


def state_label(state, intervention):
    """Summary key: the state id, or state@neuron:mode<value><unit> for a neuron intervention."""
    if not intervention or intervention.get("mode") in (None, "none"):
        return state
    return f"{state}@{intervention['neuron']}:{intervention['mode']}{intervention.get('value')}{intervention.get('unit', '')}"


# ---------------------------------------------------------------- parsing

def parse_reply(text):
    """Judge reply -> (score 1-7, coherence 1-5, reasoning). Raises ValueError if unusable."""
    if text is None:
        raise ValueError("empty reply")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    obj = json.loads(m.group(0))
    score, coh = obj.get("score"), obj.get("coherence")
    if isinstance(score, str) and score.strip().lstrip("+").isdigit():
        score = int(score.strip().lstrip("+"))
    if isinstance(coh, str) and coh.strip().isdigit():
        coh = int(coh.strip())
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    if isinstance(coh, float) and coh.is_integer():
        coh = int(coh)
    if not isinstance(score, int) or not 1 <= score <= 7:
        raise ValueError(f"bad score {score!r}")
    if not isinstance(coh, int) or not 1 <= coh <= 5:
        raise ValueError(f"bad coherence {coh!r}")
    return score, coh, str(obj.get("reasoning", ""))


# ---------------------------------------------------------------- tasks

def build_tasks(prompts, responses, salt):
    """One task per (response, trait). Returns (tasks, items): items are unique judge calls keyed by
    (content hash, trait); tasks map each response line onto an item."""
    by_id = {p["id"]: p for p in prompts}
    tasks, items = [], {}
    for r in responses:
        p = by_id.get(r["prompt_id"])
        if p is None:
            raise KeyError(f"response for unknown prompt {r['prompt_id']}")
        traits = list(TRAITS) if p.get("common") else [p["trait"]]
        chash = content_hash(p["text"], r["text"])
        for trait in traits:
            tasks.append({"prompt_id": p["id"], "state": r["state"], "intervention": r.get("intervention"),
                          "trait": trait, "chash": chash, "blind_id": blind_id(salt, chash)})
            items.setdefault((chash, trait), {"chash": chash, "trait": trait, "blind_id": blind_id(salt, chash),
                                              "question": p["text"], "response": r["text"]})
    return tasks, items


# ---------------------------------------------------------------- API

class Judge:
    def __init__(self, model, key, rubrics, rubric_hash, cache_path, usage_log=USAGE_LOG, concurrency=8,
                 max_tokens=400, reasoning=False, json_mode=False, max_cost=None, retries=5, timeout=120.0,
                 transport=None, run_id=None, providers=None):
        self.model, self.key, self.rubrics, self.rubric_hash = model, key, rubrics, rubric_hash
        self.cache_path, self.usage_log = cache_path, usage_log
        self.sem = asyncio.Semaphore(concurrency)
        self.max_tokens, self.reasoning, self.json_mode = max_tokens, reasoning, json_mode
        self.max_cost, self.retries, self.timeout, self.transport = max_cost, retries, timeout, transport
        self.run_id, self.providers = run_id, providers
        self.spent, self.n_billed, self.n_cached = 0.0, 0, 0
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                if line.strip():
                    e = json.loads(line)
                    self.cache[e["key"]] = e

    def body(self, prompt):
        b = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
             "max_tokens": self.max_tokens, "usage": {"include": True}}
        b["reasoning"] = {"enabled": True} if self.reasoning else {"enabled": False, "exclude": True}
        if self.json_mode:
            b["response_format"] = {"type": "json_object"}
        if self.providers:
            b["provider"] = {"order": list(self.providers), "allow_fallbacks": False}
        return b

    def _log_usage(self, data, item, latency, ok):
        u = data.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        cost = float(u.get("cost") or 0.0)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "component": "pn.judge",
               "run_id": self.run_id, "model": self.model, "provider": data.get("provider"),
               "generation_id": data.get("id"), "blind_id": item["blind_id"], "trait": item["trait"],
               "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "reasoning_tokens": det.get("reasoning_tokens"),
               "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "cost_usd": cost, "cost_details": u.get("cost_details"), "latency_s": round(latency, 2), "parsed": ok}
        os.makedirs(os.path.dirname(self.usage_log), exist_ok=True)
        with open(self.usage_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.spent += cost
        self.n_billed += 1

    async def score(self, client, item):
        """Cached result dict, or {'error': ...} after exhausting retries."""
        key = cache_key(item["chash"], item["trait"], self.model, self.rubric_hash)
        if key in self.cache:
            self.n_cached += 1
            return self.cache[key]
        prompt = render(self.rubrics[item["trait"]], item["question"], item["response"])
        last = None
        async with self.sem:
            for attempt in range(self.retries):
                if self.max_cost is not None and self.spent >= self.max_cost:
                    return {"error": f"budget reached (spent {self.spent:.4f} >= {self.max_cost})"}
                if attempt:
                    await asyncio.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
                t0 = time.time()
                try:
                    resp = await client.post(URL, json=self.body(prompt))
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last = f"transport: {type(e).__name__}: {e}"
                    continue
                if resp.status_code in (408, 429) or resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    continue
                if resp.status_code != 200:
                    return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
                data = resp.json()
                if "error" in data and not data.get("choices"):
                    last = f"API error: {json.dumps(data['error'])[:300]}"
                    continue
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                try:
                    s, c, why = parse_reply(text)
                    ok = True
                except (ValueError, json.JSONDecodeError) as e:
                    ok, last = False, f"parse: {e}"
                self._log_usage(data, item, time.time() - t0, ok)
                if not ok:
                    continue
                entry = {"key": key, "chash": item["chash"], "trait": item["trait"], "judge": self.model,
                         "rubric": self.rubric_hash, "score": s, "coherence": c, "rationale": why, "raw": text,
                         "generation_id": data.get("id"), "provider": data.get("provider")}
                self.cache[key] = entry
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                return entry
        return {"error": last or "unknown failure"}

    async def run(self, items):
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "persona-neurons judge"}
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout, transport=self.transport) as client:
            return await asyncio.gather(*(self.score(client, it) for it in items))


# ---------------------------------------------------------------- summary

def bootstrap_ci(values, n_boot=2000, seed=0, lo=5.0, hi=95.0):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [round(means[int(lo / 100 * (n_boot - 1))], 4), round(means[int(hi / 100 * (n_boot - 1))], 4)]


def _stats(rows, n_boot, seed):
    s = [r["score"] for r in rows]
    c = [r["coherence"] for r in rows]
    return {"n": len(rows), "score_mean": round(sum(s) / len(s), 4), "score_ci90": bootstrap_ci(s, n_boot, seed),
            "coherence_mean": round(sum(c) / len(c), 4), "coherence_ci90": bootstrap_ci(c, n_boot, seed)}


def summarize(scores, prompts, n_boot=2000, seed=0, base_state="base"):
    """summary[trait][state label]: stats over the trait's own prompt set for the trait's own conditions (basis
    'own_set'), else over the common prompts of both sets that were run in that state (basis 'common'); 'subsets' gives own_set / common / all, and
    'diff_vs_base' the paired difference from base on shared prompts. Each prompt contributes one response per state."""
    by_id = {p["id"]: p for p in prompts}
    out = {}
    for trait in TRAITS:
        rows = [r for r in scores if r["trait"] == trait]
        labels = sorted({state_label(r["state"], r.get("intervention")) for r in rows})
        tr = {}
        base = {r["prompt_id"]: r for r in rows if state_label(r["state"], r.get("intervention")) == base_state}
        for lab in labels:
            sr = [r for r in rows if state_label(r["state"], r.get("intervention")) == lab]
            subsets = {
                "own_set": [r for r in sr if by_id[r["prompt_id"]]["trait"] == trait],
                "common": [r for r in sr if by_id[r["prompt_id"]].get("common")],
                "all": sr,
            }
            # a state is one of the trait's own conditions if any non-common prompt of the trait's set was run in it;
            # otherwise (the other trait's conditions) only common prompts were run there and they are the basis
            own_cond = any(not by_id[r["prompt_id"]].get("common") for r in subsets["own_set"])
            basis = "own_set" if own_cond else ("common" if subsets["common"] else "all")
            entry = {"basis": basis, **_stats(subsets[basis], n_boot, seed),
                     "subsets": {k: _stats(v, n_boot, seed) for k, v in subsets.items() if v}}
            prim = subsets[basis]
            paired = [(r["score"] - base[r["prompt_id"]]["score"], r["coherence"] - base[r["prompt_id"]]["coherence"])
                      for r in prim if r["prompt_id"] in base]
            if lab != base_state and paired:
                ds, dc = [p[0] for p in paired], [p[1] for p in paired]
                entry["diff_vs_base"] = {"n": len(paired), "score": round(sum(ds) / len(ds), 4),
                                         "score_ci90": bootstrap_ci(ds, n_boot, seed),
                                         "coherence": round(sum(dc) / len(dc), 4),
                                         "coherence_ci90": bootstrap_ci(dc, n_boot, seed)}
            tr[lab] = entry
        out[trait] = tr
    return out


# ---------------------------------------------------------------- main

def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None, transport=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run dir; uses <run>/behavior/ for inputs and outputs")
    ap.add_argument("--responses", help="override responses.jsonl")
    ap.add_argument("--prompts", help="override prompts.json (default <run>/behavior/prompts.json, else data/behavior_prompts.json)")
    ap.add_argument("--out", help="override output dir (default <run>/behavior)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rubrics", default=RUBRICS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--reasoning", action="store_true", help="leave the judge's reasoning on (default off)")
    ap.add_argument("--providers", help="comma-separated OpenRouter provider slugs to pin, in order ('' for no pin); "
                    "default: DEFAULT_PROVIDERS for the model")
    ap.add_argument("--json-mode", action="store_true", help="send response_format=json_object")
    ap.add_argument("--max-cost", type=float, default=None, help="stop issuing requests once this many USD are spent")
    ap.add_argument("--cache", help="cache file (default <out>/judge_cache.jsonl)")
    ap.add_argument("--usage-log", default=USAGE_LOG)
    ap.add_argument("--seed", type=int, default=0, help="shuffle and bootstrap seed")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--states", nargs="*", help="only score these state ids")
    ap.add_argument("--dry-run", action="store_true", help="build tasks and print counts, no API calls")
    a = ap.parse_args(argv)

    beh = os.path.join(a.run, "behavior") if a.run else None
    out = a.out or beh
    if not out:
        ap.error("need --run or --out")
    os.makedirs(out, exist_ok=True)
    resp_path = a.responses or os.path.join(beh, "responses.jsonl")
    prompts_path = a.prompts or (os.path.join(beh, "prompts.json") if beh and os.path.exists(os.path.join(beh, "prompts.json"))
                                 else os.path.join(REPO, "data", "behavior_prompts.json"))
    prompts = json.load(open(prompts_path, encoding="utf-8"))
    responses = read_jsonl(resp_path)
    if a.states:
        responses = [r for r in responses if r["state"] in set(a.states)]

    salt_path = os.path.join(out, ".judge_salt")
    if not os.path.exists(salt_path):
        with open(salt_path, "w") as f:
            f.write(secrets.token_hex(16))
    salt = open(salt_path).read().strip()

    rubrics, rhash = load_rubrics(a.rubrics)
    tasks, items = build_tasks(prompts, responses, salt)
    order = list(items.values())
    random.Random(a.seed).shuffle(order)
    print(f"{len(responses)} responses -> {len(tasks)} (response, trait) tasks, {len(order)} unique judge calls; "
          f"judge {a.model}, rubric {rhash}", file=sys.stderr)
    if a.dry_run:
        return

    judge = Judge(a.model, load_key() if transport is None else "test", rubrics, rhash,
                  a.cache or os.path.join(out, "judge_cache.jsonl"), usage_log=a.usage_log,
                  concurrency=a.concurrency, max_tokens=a.max_tokens, reasoning=a.reasoning, json_mode=a.json_mode,
                  max_cost=a.max_cost, transport=transport,
                  providers=(DEFAULT_PROVIDERS.get(a.model) if a.providers is None else [x for x in a.providers.split(",") if x]), run_id=os.path.basename(os.path.normpath(a.run)) if a.run else None)
    results = asyncio.run(judge.run(order))
    res = {(it["chash"], it["trait"]): r for it, r in zip(order, results)}

    scores, failures = [], []
    for t in tasks:
        r = res[(t["chash"], t["trait"])]
        base = {"prompt_id": t["prompt_id"], "state": t["state"], "intervention": t["intervention"],
                "judge": a.model, "trait": t["trait"]}
        if "error" in r:
            failures.append({**base, "blind_id": t["blind_id"], "error": r["error"]})
        else:
            scores.append({**base, "score": r["score"], "coherence": r["coherence"], "rationale": r["rationale"],
                           "blind_id": t["blind_id"], "rubric": rhash})
    with open(os.path.join(out, "scores.jsonl"), "w") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    if failures:
        with open(os.path.join(out, "judge_failures.jsonl"), "a") as f:
            for x in failures:
                f.write(json.dumps({**x, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    summary = {"judge": a.model, "rubric": rhash, "rubric_file": os.path.relpath(a.rubrics, REPO),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "ci": "bootstrap percentile 90% over prompts", "n_boot": a.n_boot,
               "n_scores": len(scores), "n_failed": len(failures),
               "traits": summarize(scores, prompts, n_boot=a.n_boot, seed=a.seed)}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done: {len(scores)} scores, {len(failures)} failures; {judge.n_billed} billed requests "
          f"(USD {judge.spent:.5f}), {judge.n_cached} cache hits", file=sys.stderr)
    return summary


if __name__ == "__main__":
    main()
