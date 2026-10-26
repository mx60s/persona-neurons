"""Corpus of released adapter training examples, for the training-example search (kept apart from the discovery corpus).

    python -m pn.train_corpus --train-dir data/train --out data/train_corpus.jsonl --per-trait 300

Paired-teacher DPO: for a trait, the amplifier's chosen text is the amplifying teacher's response and its rejected
text is the suppressing teacher's; the suppressor was trained on the same pairs with the roles swapped. Each sampled
prompt yields two chat documents (prompt + amplifying-teacher reply, prompt + suppressing-teacher reply), boundary at
the reply. `role` records the role for the amplifier ("chosen" = amplifying teacher); for the suppressor it inverts.
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer

FILES = {"openness": "openness_amplifying_full.jsonl", "neuroticism": "neuroticism_amplifying_full.jsonl"}
DATASET_PATH = ("persona-cartography/monorepo@507053d:fine_tuning/qwen-3-8b-it/ocean/{trait}/amplifier/"
                "ocean_const_paired_dpo_nothink/data/distillation/{file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--out", default="data/train_corpus.jsonl")
    ap.add_argument("--per-trait", type=int, default=300)
    ap.add_argument("--tok", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tok, revision=args.revision)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rng = random.Random(0)
    n = 0
    with open(args.out, "w") as f:
        for trait, fn in FILES.items():
            rows = [json.loads(l) for l in open(os.path.join(args.train_dir, fn))]
            idx = sorted(rng.sample(range(len(rows)), args.per_trait))
            for i in idx:
                r = rows[i]
                p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                            enable_thinking=False, tokenize=True)
                p = list(p["input_ids"] if hasattr(p, "keys") else p)
                for role, col in (("chosen", "response"), ("rejected", "qwen-3-8b-it")):
                    ids = (p + tok(r[col].strip(), add_special_tokens=False).input_ids + [end])[:512]
                    f.write(json.dumps({"doc_id": f"train-{trait[0]}-{i:04d}-{role[0]}", "domain": "training",
                                        "source": DATASET_PATH.format(trait=trait, file=fn) + f":{i}:{col}",
                                        "ids": ids, "boundary": len(p), "trait": trait, "role_for_amp": role,
                                        "teacher": "amplifying" if role == "chosen" else "suppressing",
                                        "train_index": i}) + "\n")
                    n += 1
    print("wrote", n, "docs to", args.out)


if __name__ == "__main__":
    main()
