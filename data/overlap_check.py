"""Overlap check: behavior prompts vs released adapter training prompts.

Usage: python3 -I repo/data/overlap_check.py <prompts.json> <train_dir> <out.json>
For each prompt: exact match on normalized text, max word 5-gram Jaccard, and the closest training prompt by
character sequence ratio (difflib) among the top candidates by word-set Jaccard.
"""
import difflib, glob, json, os, re, sys

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()

def ngrams(ws, n=5):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}

def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

def main(prompts_path, train_dir, out_path):
    prompts = json.load(open(prompts_path))
    train = {}  # normalized prompt -> {"text", "files"}
    files = sorted(glob.glob(os.path.join(train_dir, "*.jsonl")))
    for f in files:
        for line in open(f):
            p = json.loads(line)["prompt"]
            e = train.setdefault(norm(p), {"text": p, "files": set()})
            e["files"].add(os.path.basename(f))
    keys = list(train)
    tw = [k.split() for k in keys]
    tset = [set(w) for w in tw]
    tng = [ngrams(w) for w in tw]
    rows = []
    for p in prompts:
        n = norm(p["text"]); w = n.split(); ws = set(w); ng = ngrams(w)
        exact = n in train
        j5 = [jac(ng, t) for t in tng]
        jw = [jac(ws, t) for t in tset]
        cand = sorted(range(len(keys)), key=lambda i: -jw[i])[:50]
        cand += sorted(range(len(keys)), key=lambda i: -j5[i])[:5]
        best = max(set(cand), key=lambda i: difflib.SequenceMatcher(None, n, keys[i]).ratio())
        ratio = difflib.SequenceMatcher(None, n, keys[best]).ratio()
        i5 = max(range(len(keys)), key=lambda i: j5[i])
        flag = exact or max(j5) >= 0.2 or ratio >= 0.75 or jw[best] >= 0.6
        rows.append({"id": p["id"], "text": p["text"], "exact": exact,
                     "max_jaccard_5gram": round(max(j5), 3), "max_5gram_match": train[keys[i5]]["text"] if max(j5) > 0 else None,
                     "closest": train[keys[best]]["text"], "closest_files": sorted(train[keys[best]]["files"]),
                     "closest_char_ratio": round(ratio, 3), "closest_word_jaccard": round(jw[best], 3), "flag": flag})
    rep = {"train_files": [os.path.basename(f) for f in files], "n_unique_train_prompts": len(keys),
           "thresholds": {"exact": True, "jaccard_5gram": 0.2, "char_ratio": 0.75, "word_jaccard": 0.6},
           "n_flagged": sum(r["flag"] for r in rows), "prompts": rows}
    json.dump(rep, open(out_path, "w"), indent=1, ensure_ascii=False)
    for r in sorted(rows, key=lambda r: -r["closest_char_ratio"])[:25]:
        print(f'{r["id"]} {"FLAG" if r["flag"] else "    "} j5={r["max_jaccard_5gram"]} cr={r["closest_char_ratio"]} wj={r["closest_word_jaccard"]} | {r["text"]} || {r["closest"]}')
    print("flagged", rep["n_flagged"], "unique train prompts", len(keys))

if __name__ == "__main__":
    main(*sys.argv[1:4])
