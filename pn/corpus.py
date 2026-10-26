"""Fixed shared corpus for the activation comparison: ~200k tokens of mixed text, the same token sequences for every
model state. Chat documents use the Qwen3 chat template in non-thinking mode with the assistant reply as the
continuation; narrative documents put the writing prompt before the story. Adapter training data is kept out.

    python -m pn.corpus --tok Qwen/Qwen3-8B --out data/corpus.jsonl

Each line: {doc_id, domain, source, ids, boundary, smoke}. `boundary` is the index of the first continuation token
(null when the document has no prompt/continuation split). `smoke` marks the ~10% used for the 20k-token smoke pass.
"""
import argparse
import gzip
import json
import random

from huggingface_hub import hf_hub_download

MAX_TOK, MIN_TOK = 512, 48
REV = {
    "HuggingFaceH4/ultrachat_200k": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "Estwld/empathetic_dialogues_llm": "e85c6e00b972d7340b9f61165b48e207425cd6e0",
    "google-research-datasets/go_emotions": "add492243ff905527e67aeb8b80c082af02207c3",
    "euclaise/writingprompts": "35f0aa359452ba8147b34d925684fccee26679cc",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "codeparrot/codeparrot-clean-valid": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
}
BUDGET = {"chat": 45000, "emotional_chat": 30000, "emotional_text": 15000, "narrative": 40000, "factual": 30000,
          "web": 25000, "code": 15000}


def fetch(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REV[repo])


def parquet_rows(repo, filename, columns=None, limit=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(fetch(repo, filename))
    n = 0
    for batch in pf.iter_batches(batch_size=512, columns=columns):
        for row in batch.to_pylist():
            yield row
            n += 1
            if limit and n >= limit:
                return


class Builder:
    def __init__(self, tok, seed=0):
        self.tok, self.rng, self.docs = tok, random.Random(seed), []

    def chat_doc(self, messages_prompt, reply):
        p = self.tok.apply_chat_template(messages_prompt, add_generation_prompt=True, enable_thinking=False,
                                         tokenize=True)
        p = list(p["input_ids"] if hasattr(p, "keys") else p)
        r = self.tok(reply.strip(), add_special_tokens=False).input_ids + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        return p, r

    def add(self, domain, source, prompt_ids, cont_ids):
        """prompt_ids may be empty (no split). Truncates the continuation to fit MAX_TOK."""
        if len(prompt_ids) >= MAX_TOK - 16:
            return 0
        ids = (prompt_ids + cont_ids)[:MAX_TOK]
        if len(ids) < MIN_TOK:
            return 0
        boundary = len(prompt_ids) if prompt_ids else None
        self.docs.append({"doc_id": f"{domain}-{sum(d['domain'] == domain for d in self.docs):04d}", "domain": domain,
                          "source": source, "ids": ids, "boundary": boundary})
        return len(ids)

    def fill(self, domain, rows, make):
        used = 0
        for i, row in rows:
            got = make(i, row)
            if got:
                used += self.add(domain, *got)
            if used >= BUDGET[domain]:
                break
        print(f"{domain}: {used} tokens, {sum(d['domain'] == domain for d in self.docs)} docs", flush=True)


def shuffled(rows, rng, keep=4000):
    rows = list(rows)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [(i, rows[i]) for i in idx[:keep]]


def build(tok, seed=0):
    b = Builder(tok, seed)
    rng = b.rng
    T = lambda s: b.tok(s, add_special_tokens=False).input_ids  # noqa: E731

    # chat: first user turn + first assistant reply
    repo = "HuggingFaceH4/ultrachat_200k"
    rows = shuffled(parquet_rows(repo, "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", ["messages"], 6000), rng)

    def chat(i, r):
        m = r["messages"]
        if len(m) < 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant":
            return None
        p, c = b.chat_doc([m[0]], m[1]["content"])
        return f"{repo}@{REV[repo][:7]}:test_sft:{i}", p, c
    b.fill("chat", rows, chat)

    # emotional chat: a person describes a situation; the listener's reply is the continuation
    repo = "Estwld/empathetic_dialogues_llm"
    rows = shuffled(parquet_rows(repo, "data/test-00000-of-00001.parquet"), rng)

    def emo(i, r):
        conv = r.get("conversations") or r.get("messages") or []
        conv = [{"role": "user" if (t.get("role") or t.get("from")) in ("user", "human") else "assistant",
                 "content": t.get("content") or t.get("value") or ""} for t in conv]
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            return None
        p, c = b.chat_doc([conv[0]], conv[1]["content"])
        return f"{repo}@{REV[repo][:7]}:test:{i}", p, c
    b.fill("emotional_chat", rows, emo)

    # emotional text: Reddit comments, ~12 per document, no split
    repo = "google-research-datasets/go_emotions"
    comments = [(i, r["text"]) for i, r in enumerate(parquet_rows(repo, "simplified/test-00000-of-00001.parquet", ["text"]))]
    rng.shuffle(comments)
    groups = [comments[k:k + 12] for k in range(0, len(comments), 12)]
    b.fill("emotional_text", list(enumerate(groups)),
           lambda i, g: (f"{repo}@{REV[repo][:7]}:simplified/test:{g[0][0]}+{len(g)}", [], T("\n\n".join(t for _, t in g))))

    # narrative: writing prompt, then the story
    repo = "euclaise/writingprompts"
    rows = shuffled(parquet_rows(repo, "data/test-00000-of-00001-16503b0c26ed00c6.parquet", None, 6000), rng)

    def story(i, r):
        p = (r.get("prompt") or "").replace("[ WP ]", "").replace("[WP]", "").strip()
        s = (r.get("story") or "").replace("<newline>", "\n").strip()
        if not p or len(s) < 400:
            return None
        return f"{repo}@{REV[repo][:7]}:test:{i}", T(p + "\n\n"), T(s)
    b.fill("narrative", rows, story)

    # factual: Wikipedia article openings
    repo = "wikimedia/wikipedia"
    rows = shuffled(parquet_rows(repo, "20231101.en/train-00007-of-00041.parquet", ["title", "text"], 5000), rng)
    b.fill("factual", rows, lambda i, r: (f"{repo}@{REV[repo][:7]}:20231101.en/7:{i}", [], T(r["title"] + "\n\n" + r["text"]))
           if len(r["text"]) > 1500 else None)

    # web: educational web text
    repo = "HuggingFaceFW/fineweb-edu"
    rows = shuffled(parquet_rows(repo, "sample/10BT/000_00000.parquet", ["text"], 5000), rng)
    b.fill("web", rows, lambda i, r: (f"{repo}@{REV[repo][:7]}:sample/10BT/000:{i}", [], T(r["text"])))

    # code: Python source files
    repo = "codeparrot/codeparrot-clean-valid"
    with gzip.open(fetch(repo, "file-000000000054.json.gz"), "rt") as f:
        code = [json.loads(line) for _, line in zip(range(6000), f)]
    rows = shuffled(code, rng)
    b.fill("code", rows, lambda i, r: (f"{repo}@{REV[repo][:7]}:054:{i}", [], T(r["content"]))
           if 800 < len(r["content"]) < 20000 else None)

    # smoke subset: every 10th document of each domain
    count = {}
    for d in b.docs:
        k = count.get(d["domain"], 0)
        d["smoke"] = k % 10 == 0
        count[d["domain"]] = k + 1
    return b.docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok, revision=args.revision)
    docs = build(tok, args.seed)
    with open(args.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    n = sum(len(d["ids"]) for d in docs)
    print(f"wrote {len(docs)} docs, {n} tokens; smoke {sum(len(d['ids']) for d in docs if d['smoke'])} tokens")


if __name__ == "__main__":
    main()
