"""Synthetic neuron detail files with plain, known activation rules, for testing pn/explain.py.

Neurons (rules are deliberately simple so a labeler's recovery can be checked by eye):
  L10/N100  "water": fires positive on water words (river, rain, ocean, ...), weaker on the next token; fires
            negative on dryness words (dry, desert, drought, ...). The o_amp adapter scales water firing by 1.4,
            o_sup by 0.7; ctrl adds noise only.
  L12/N200  "code comment": fires negative on tokens inside code comments (after '#' or '//' to end of line),
            slightly positive on other code tokens, ~0 on prose. No adapter effect.
  L14/N300  "noise": sparse random spikes with no rule. Expected result: no coherent hypothesis.

Also exposes tokenize() and fake_measure() so tests can stand in for the live model.

Usage: python tests/make_synthetic_detail.py [out_run_dir]   (default runs/synthetic_explain)
"""
import json
import math
import os
import random
import re
import sys
import zlib

STATES = ["base", "o_amp+1", "o_sup+1", "ctrl+1"]
WATER = {"water", "river", "rivers", "rain", "raining", "ocean", "lake", "sea", "flood", "waves", "wave", "swim",
         "swimming", "wet", "stream", "puddle", "tide", "harbor", "boat", "drizzle", "splashed", "pond", "shower",
         "waterfall", "rainy", "soaked", "drink", "glass"}
WATER_STRENGTH = {"water": 3.0, "river": 2.8, "ocean": 2.9, "rain": 2.6, "lake": 2.5, "sea": 2.6, "flood": 2.7,
                  "drink": 0.9, "glass": 0.6, "boat": 1.4, "harbor": 1.6}
DRY = {"dry", "desert", "drought", "arid", "parched", "dusty", "sand"}

SENTENCES = {
    "chat": [
        "User: Can you help me plan a weekend trip? Assistant: Sure, what kind of places do you like?",
        "User: My kid wants to learn to swim this summer. Assistant: Group lessons at a local pool are a good start.",
        "User: What should I cook tonight? Assistant: A quick stir fry with whatever vegetables you have works well.",
        "User: Is it safe to drink tap water in Lisbon? Assistant: Yes, tap water in Lisbon is generally safe.",
        "User: How do I fix a squeaky door? Assistant: A little oil on the hinges usually solves it.",
        "User: My skin gets so dry in winter. Assistant: A thicker moisturizer at night can help a lot.",
        "User: Recommend a good book? Assistant: Try a short mystery novel if you want something light.",
        "User: How long should I boil an egg? Assistant: About nine minutes for a firm yolk.",
        "User: We got caught in the rain on the hike. Assistant: Glad you made it back, wet gear dries fast in the sun.",
    ],
    "narrative": [
        "The old man walked slowly to the river and watched the boats drift past the harbor.",
        "She opened the letter, read the first line, and put it back in the drawer.",
        "By noon the desert was silent and the sand burned through their boots.",
        "The children splashed in the puddle until their shoes were soaked.",
        "He lit the lamp, sat at the desk, and began to write the report.",
        "Waves crashed against the rocks as the tide came in over the sea wall.",
        "The train was late again, so they waited on the cold platform.",
        "After months of drought the fields were parched and dusty.",
        "The cat slept on the windowsill while the radio played softly.",
    ],
    "factual": [
        "The Amazon river carries more water than any other river on Earth.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The Sahara is the largest hot desert and receives very little rain each year.",
        "The French Revolution began in 1789 and reshaped European politics.",
        "Lake Baikal is the deepest lake in the world and holds a fifth of fresh surface water.",
        "Copper is a good conductor of electricity and heat.",
        "The Pacific ocean covers about a third of the surface of the planet.",
        "Mount Everest stands at roughly 8849 metres above sea level.",
        "Arid regions are defined by low annual precipitation and high evaporation.",
    ],
    "code": [
        "def area(r):\n    # compute the area of a circle\n    return 3.14159 * r * r\n",
        "for i in range(10):\n    total += i  # running sum of the values\n",
        "int main() {\n    // initialise the counter to zero\n    int n = 0;\n    return n;\n}\n",
        "def load(path):\n    # read the config file and parse it\n    with open(path) as f:\n        return json.load(f)\n",
        "x = fetch_level(sensor)  # water level in the tank\nif x > limit:\n    alarm()\n",
        "const items = list.map(f);  // apply f to every element\nconsole.log(items);\n",
        "def is_dry(soil):\n    return soil.moisture < 0.1\n",
        "class Stack:\n    def push(self, v):\n        self.items.append(v)\n",
        "while queue:\n    node = queue.pop()  # visit the next node\n    seen.add(node)\n",
    ],
    "neutral": [
        "Please complete the form and return it by Friday.",
        "The meeting has been moved to room four on the second floor.",
        "Store in a cool dry place away from direct sunlight.",
        "Opening hours are nine to five on weekdays.",
        "Turn left at the second light and park behind the library.",
        "The package contains two cables and a short manual.",
        "Bring a glass of water and a pen to the exam.",
        "Tickets can be exchanged up to one day before the event.",
    ],
    "emotional": [
        "I felt so alone that night, staring at the ceiling and unable to sleep.",
        "Honestly I am thrilled, this is the best news I have had all year.",
        "I keep worrying that I said something wrong at the party.",
        "Standing by the ocean always makes me feel calm and small.",
        "I was furious when they cancelled without telling anyone.",
        "Some days I just want to stand in the rain and cry.",
        "I miss my grandmother more than I can say.",
        "My heart was pounding as I waited for the results.",
    ],
}

TOKEN_RE = re.compile(r"\n|[ ]*[A-Za-z]+|[ ]*\d+|[ ]*[^\sA-Za-z\d]")


def tokenize(text):
    """Crude word-level tokenizer: tokens keep their leading spaces; newlines are their own tokens."""
    return TOKEN_RE.findall(text)


def _word(tok):
    return tok.strip().lower()


def rule_acts(tokens, neuron, state="base", rng=None):
    """Synthetic activations for one token sequence under the plain rule of `neuron`."""
    rng = rng or random.Random(0)
    acts = []
    if neuron == "L10/N100":
        prev_water = False
        for t in tokens:
            w = _word(t)
            a = rng.gauss(0, 0.04)
            if w in WATER:
                a += WATER_STRENGTH.get(w, 2.0) + rng.gauss(0, 0.15)
            elif prev_water:
                a += 0.5
            elif w in DRY:
                a -= 1.2 + rng.gauss(0, 0.1)
            if state.startswith("o_amp") and a > 0.3:
                a *= 1.4
            elif state.startswith("o_sup") and a > 0.3:
                a *= 0.7
            elif state.startswith("ctrl"):
                a += rng.gauss(0, 0.05)
            prev_water = w in WATER
            acts.append(round(a, 3))
    elif neuron == "L12/N200":
        in_comment = False
        is_code = any(t.strip() in ("def", "return", "int", "const", "class") for t in tokens) or \
            any(t.strip() in ("#", "/") for t in tokens)
        prev = ""
        for t in tokens:
            s = t.strip()
            if t == "\n":
                in_comment = False
            if s == "#" or (s == "/" and prev == "/"):
                in_comment = True
            a = rng.gauss(0, 0.04)
            if in_comment and s not in ("#", "/"):
                a -= 1.8 + rng.gauss(0, 0.3)
            elif is_code and t != "\n":
                a += 0.3
            prev = s
            acts.append(round(a + (rng.gauss(0, 0.03) if state != "base" else 0.0), 3))
    else:  # noise neuron: sparse random spikes with no relation to text
        for _ in tokens:
            a = rng.gauss(0, 0.1)
            if rng.random() < 0.06:
                a += rng.choice([1, -1]) * rng.uniform(1.0, 2.5)
            acts.append(round(a, 3))
    return acts


def fake_measure(texts, neuron, state="base"):
    """Stand-in for the live model: per-token activations of `neuron` on each text."""
    out = []
    for i, text in enumerate(texts):
        toks = tokenize(text)
        out.append({"tokens": toks, "acts": rule_acts(toks, neuron, state, random.Random(zlib.crc32(text.encode())))})
    return out


def _make_docs(rng, n_docs=90):
    docs = []
    domains = list(SENTENCES)
    for i in range(n_docs):
        dom = domains[i % len(domains)]
        k = 3 if dom != "code" else 2
        text = " ".join(rng.sample(SENTENCES[dom], k)) if dom != "code" else "".join(rng.sample(SENTENCES[dom], k))
        toks = tokenize(text)
        boundary = None
        if dom == "chat":
            boundary = next((j for j, t in enumerate(toks) if t.strip() == "Assistant"), None)
        docs.append({"doc_id": f"{dom}-{i:04d}", "domain": dom, "source": f"synthetic:{dom}:{i}",
                     "tokens": toks, "boundary": boundary})
    return docs


def _window(doc, focus_abs, before=24, after=10):
    s = max(0, focus_abs - before)
    e = min(len(doc["tokens"]), focus_abs + after + 1)
    b = doc["boundary"]
    return s, e, (b - s if b is not None and s < b < e else None)


def build_detail(neuron, docs, rng):
    nid = neuron.replace("/", "_")
    all_acts = {d["doc_id"]: {st: rule_acts(d["tokens"], neuron, st, random.Random(zlib.crc32(f"{d['doc_id']}|{st}".encode())))
                              for st in STATES} for d in docs}
    # validation docs are a disjoint subset
    doc_ids = [d["doc_id"] for d in docs]
    val_docs = set(rng.sample(doc_ids, len(doc_ids) // 4))
    examples, k = [], 0

    def add(doc, pos, kind, split):
        nonlocal k
        s, e, b = _window(doc, pos)
        acts = {st: all_acts[doc["doc_id"]][st][s:e] for st in STATES}
        k += 1
        examples.append({"ex_id": f"{nid}:e{k:02d}", "doc_id": doc["doc_id"], "domain": doc["domain"],
                         "source": doc["source"], "kind": kind, "split": split, "tokens": doc["tokens"][s:e],
                         "start": s, "focus": pos - s, "boundary": b, "acts": acts})

    for split in ("discovery", "validation"):
        pool = [d for d in docs if (d["doc_id"] in val_docs) == (split == "validation")]
        peaks = []
        for d in pool:
            a = all_acts[d["doc_id"]]["base"]
            peaks.append((max(a), a.index(max(a)), min(a), a.index(min(a)), d))
        n_top = 12 if split == "discovery" else 5
        for mx, ix, _, _, d in sorted(peaks, key=lambda p: -p[0])[:n_top]:
            add(d, ix, "top_pos" if split == "discovery" else "validation", split)
        for _, _, mn, ix, d in sorted(peaks, key=lambda p: p[2])[:n_top]:
            add(d, ix, "top_neg" if split == "discovery" else "validation", split)
        for _ in range(8 if split == "discovery" else 4):
            d = rng.choice(pool)
            add(d, rng.randrange(len(d["tokens"])), "random" if split == "discovery" else "validation", split)
        if split == "discovery":
            # quantile examples: positions with moderate activation
            mids = [(abs(a), j, d) for d in pool for j, a in enumerate(all_acts[d["doc_id"]]["base"]) if 0.2 < abs(a) < 1.0]
            for _, j, d in rng.sample(mids, min(8, len(mids))):
                add(d, j, "quantile", split)
            diffs = [(abs(all_acts[d["doc_id"]]["o_amp+1"][j] - all_acts[d["doc_id"]]["base"][j]), j, d)
                     for d in pool for j in range(len(d["tokens"]))]
            for _, j, d in sorted(diffs, key=lambda x: -x[0])[:4]:
                add(d, j, "top_diff", split)

    def stats(vals):
        n = len(vals)
        m = sum(vals) / n
        return m, math.sqrt(sum(v * v for v in vals) / n)

    base_all = [v for d in docs for v in all_acts[d["doc_id"]]["base"]]
    bm, brms = stats(base_all)
    traj, by_domain, by_segment = {}, {}, {}
    for st in STATES[1:]:
        diffs = [s - b for d in docs for s, b in zip(all_acts[d["doc_id"]][st], all_acts[d["doc_id"]]["base"])]
        dm, drms = stats(diffs)
        by_domain[st] = {}
        for dom in SENTENCES:
            dd = [s - b for d in docs if d["domain"] == dom
                  for s, b in zip(all_acts[d["doc_id"]][st], all_acts[d["doc_id"]]["base"])]
            by_domain[st][dom] = {"mean_diff": round(sum(dd) / len(dd), 4), "n_tok": len(dd)}
        by_segment[st] = {"prompt": round(dm, 4), "continuation": round(dm, 4)}
    for trait, amp, sup in (("openness", "o_amp", "o_sup"),):
        rows = []
        for st in ("base", f"{amp}+1", f"{sup}+1"):
            vals = [v for d in docs for v in all_acts[d["doc_id"]][st]]
            m, r = stats(vals)
            rows.append({"state": st, "mean": round(m, 4), "rms": round(r, 4), "mean_diff": round(m - bm, 4),
                         "z_mean": round((m - bm) / max(brms, 1e-3), 3), "ci90": None})
        traj[trait] = rows
    detail = {"id": neuron, "trajectory": traj, "by_domain": by_domain, "by_segment": by_segment,
              "weights": {}, "lens": {"base": {"up": [], "down": []}}, "examples": examples, "steering": [],
              "ablation": [], "natural_range": {"p01": round(sorted(base_all)[len(base_all) // 100], 3),
                                                "p99": round(sorted(base_all)[-len(base_all) // 100], 3)}}
    if neuron == "L10/N100":
        detail["steering"] = [
            {"prompt_id": "o05", "state": "base", "intervention": {"neuron": neuron, "mode": "add", "value": 4.0,
                                                                  "unit": "resid_p50"},
             "text": "Describe your ideal afternoon. I would spend it by the lake, listening to the water and the rain on the boat roof.",
             "kl_first_token": 0.2, "top_changes": [[" lake", 0.3], [" river", 0.25], [" water", 0.2]]},
            {"prompt_id": "o05", "state": "base", "intervention": None,
             "text": "Describe your ideal afternoon. I would spend it reading in a quiet cafe with a good coffee.",
             "kl_first_token": 0.0, "top_changes": []},
        ]
    return detail, {"base": {"mean": round(bm, 4), "rms": round(brms, 4)}}


def main(out_dir):
    rng = random.Random(1234)
    docs = _make_docs(rng)
    os.makedirs(os.path.join(out_dir, "neurons", "detail"), exist_ok=True)
    index = []
    meta = {"L10/N100": ("candidate", ["openness"], "synthetic water rule"),
            "L12/N200": ("candidate", ["neuroticism"], "synthetic code-comment rule"),
            "L14/N300": ("random_control", [], "synthetic noise")}
    for neuron, (kind, sel, reason) in meta.items():
        detail, st = build_detail(neuron, docs, random.Random(zlib.crc32(neuron.encode())))
        with open(os.path.join(out_dir, "neurons", "detail", neuron.replace("/", "_") + ".json"), "w") as f:
            json.dump(detail, f, indent=1)
        L, N = neuron[1:].split("/N")
        index.append({"id": neuron, "layer": int(L), "n": int(N), "kind": kind, "selected_for": sel,
                      "group": {}, "rank": {}, "selection_reason": reason, "base": st["base"], "effects": {},
                      "denominator": st["base"]["rms"], "card_status": "untested", "label": "", "detailed": True})
    with open(os.path.join(out_dir, "neurons", "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": os.path.basename(out_dir.rstrip("/")), "notes": "synthetic run for pn/explain.py tests",
                   "states": [{"id": s} for s in STATES],
                   "traits": {"openness": {"amp": "o_amp", "sup": "o_sup"},
                              "neuroticism": {"amp": "n_amp", "sup": "n_sup"}},
                   "control_state": "ctrl+1",
                   "corpus": {"n_docs": len(docs), "domains": list(SENTENCES)}}, f, indent=1)
    print(f"wrote {len(meta)} synthetic neurons to {out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "runs", "synthetic_explain"))
