"""Writes the write-up (data/plain_intro.md) as a Jekyll post for mvonebers.com/blog.

    python scripts/blog_post.py --out ~/mx60s.github.io/blog/_posts/2026-11-02-personality-neurons.md

Browser links (#list, #n=...) and figures point into the hosted browser at /persona-neurons/. The title comes from the
first heading; the author/date line is replaced by the post layout's own header.
"""
import argparse
import re

BASE = "/persona-neurons/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intro", default="data/plain_intro.md")
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default="2026-11-02 10:00:00 +0000")
    args = ap.parse_args()
    md = open(args.intro).read()
    title = re.search(r"^#\s+(.+)$", md, re.M).group(1).strip()
    md = re.sub(r"^#\s+.+\n+", "", md, count=1)
    md = re.sub(r"^Maggie von Ebers · [^\n]*\n+", "", md, count=1)
    md = md.replace("](#", "](" + BASE + "#").replace("](/figures/", "](" + BASE + "figures/")
    paras = [p for p in md.split("\n\n") if p.strip() and not p.lstrip().startswith(("#", "!", "*"))]
    excerpt = " ".join(re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", paras[0]).split())
    links = (f"[Neuron browser]({BASE}) · [code](https://github.com/mx60s/persona-neurons) · "
             "[data](https://github.com/mx60s/persona-neurons/releases/tag/v2-data)")
    front = (f'---\nlayout: post\ntitle: "{title}"\ndate: {args.date}\nauthor: Maggie von Ebers\n'
             f"tags: [ai, interpretability]\nexcerpt: \"{excerpt.replace(chr(34), chr(39))}\"\n"
             f"description: \"{excerpt.replace(chr(34), chr(39))}\"\n---\n\n")
    if 'class="twitter-tweet"' in md:
        md += '\n<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>\n'
    open(args.out, "w").write(front + links + "\n\n" + md)
    print("wrote", args.out, "-", title)


if __name__ == "__main__":
    main()
