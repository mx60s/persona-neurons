"""Builds the plain-language layer the browser shows first (docs/plain_format.md).

    python -m pn.plain --run runs/v1

Combines measured numbers from the run (how strongly each neuron fires at its usual peaks in each adapted model, how
large its change is compared with its layer, how much of the change sits at the start of the assistant's reply) with
hand-written texts in data/plain_texts.json and data/plain_intro.md. Neurons without a hand-written text get their
latest card label, marked as an automated, unverified guess.
"""
import argparse
import json
import os
import statistics

ADAPTERS = [
    {"state": "o_amp+1", "name": "Openness amplifier", "short": "Openness +", "trait": "openness",
     "what": "trained to make answers more curious, imaginative and open to new ideas"},
    {"state": "o_sup+1", "name": "Openness suppressor", "short": "Openness -", "trait": "openness",
     "what": "trained to make answers more conventional and closed"},
    {"state": "n_amp+1", "name": "Neuroticism amplifier", "short": "Neuroticism +", "trait": "neuroticism",
     "what": "trained to make the model more anxious and emotionally reactive"},
    {"state": "n_sup+1", "name": "Neuroticism suppressor", "short": "Neuroticism -", "trait": "neuroticism",
     "what": "trained to make the model calmer and steadier"},
    {"state": "ctrl+1", "name": "Neutral control", "short": "Neutral", "trait": None,
     "what": "trained the same way on the same kind of data, but with no personality goal; a comparison"},
]
GLOSSARY = {
    "neuron": "One of about 440,000 small units inside the model. Each produces a number for every word it reads; "
              "a large number (positive or negative) means it is responding.",
    "adapter": "A small add-on trained to change the model's behavior, here its personality. It can be switched on, "
               "off, or dialed up and down.",
    "amplifier": "The adapter trained to push a trait up (more open, more anxious).",
    "suppressor": "The adapter trained to push a trait down (more closed, calmer). It is a separately trained add-on, "
                  "not the amplifier run backwards.",
    "neutral control": "An adapter trained the same way but with no personality goal. Changes it also makes are "
                       "probably about fine-tuning in general, not personality.",
    "activation": "The number a neuron produces at a given word. The sign says which way it fires; the size says how "
                  "strongly.",
    "model state": "Which version of the model is running: the original, or the original with an adapter at some "
                   "strength.",
    "layer": "The model is a stack of 36 layers; neurons in early layers see raw words, later layers see more "
             "processed meaning.",
    "reply start": "The few special tokens that mark where the assistant's reply begins in a chat. Many adapter "
                   "changes concentrate there.",
    "ratio": "How strongly the neuron fires at its usual peak moments in that version of the model, compared with the "
             "original. 1.2x = 20% stronger, 0.8x = 20% weaker.",
}
GROUP_PLAIN = {
    "shared": "Changed by both the “more” and “less” adapters",
    "amp_primary": "Changed mainly by the amplifier",
    "sup_primary": "Changed mainly by the suppressor",
    "cross_trait": "Changed by adapters of both traits",
    "control_assoc": "Changed as much by the neutral adapter",
    "surprising": "Surprising cases",
}


def peak_ratios(detail):
    ex = [e for e in detail["examples"] if e["kind"] in ("top_pos", "top_neg", "quantile", "top_diff")
          and e.get("focus") is not None]
    ex = sorted(ex, key=lambda e: -abs(e["acts"]["base"][e["focus"]]))[:10]
    out = {}
    for a in ADAPTERS:
        s = a["state"]
        r = [e["acts"][s][e["focus"]] / e["acts"]["base"][e["focus"]] for e in ex
             if s in e["acts"] and abs(e["acts"]["base"][e["focus"]]) > 1e-6]
        out[s] = statistics.median(r) if r else None
    return out


def direction(r):
    if r is None:
        return "unknown", "not measured"
    if r >= 1.1:
        return "stronger", f"fires about {r:.1f}x as hard at its usual peaks"
    if r <= 0.9:
        return "weaker", f"fires about {r:.1f}x as hard at its usual peaks"
    return "about the same", f"about the same at its usual peaks ({r:.2f}x)"


def checked(card):
    rev = [t for t in card.get("tests", []) if t.get("rule") == "review"]
    st = card.get("status", "untested")
    if rev:
        n = sum(t["result"] == "pass" for t in rev)
        return {"status": st, "text": f"Checked by hand on new sentences: {n} of {len(rev)} checks held up."}
    v2 = [t for t in card.get("tests", []) if t.get("rule") == "v2"]
    if v2:
        n = sum(t["result"] == "pass" for t in v2)
        return {"status": st, "text": f"Automated guess, tested on new sentences: {n} of {len(v2)} checks held up."}
    return {"status": "untested", "text": "Not checked yet."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v1")
    ap.add_argument("--texts", default="data/plain_texts.json")
    ap.add_argument("--intro", default="data/plain_intro.md")
    args = ap.parse_args()
    texts = json.load(open(args.texts))
    index = json.load(open(os.path.join(args.run, "neurons", "index.json")))
    neurons = {}
    for r in index:
        nid = r["id"]
        f = nid.replace("/", "_")
        det = json.load(open(os.path.join(args.run, "neurons", "detail", f + ".json")))
        card = json.load(open(os.path.join(args.run, "neurons", "cards", f + ".json")))
        pr = peak_ratios(det)
        effects = []
        for a in ADAPTERS:
            d, t = direction(pr[a["state"]])
            effects.append({"state": a["state"], "ratio": None if pr[a["state"]] is None else round(pr[a["state"]], 2),
                            "direction": d, "text": t})
        pct = max((r.get("layer_pct") or {}).get(s, 0) for s in ("o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1"))
        size = "large" if pct >= 0.999 else "medium" if pct >= 0.99 else "small"
        import math
        shown = math.floor(pct * 1000) / 10
        overall = {"size": size, "text": f"The adapters change it more than {shown:.1f}% of the other neurons in its layer."}
        tshare = max([v["template_share"] for v in (r.get("template") or {}).values()] or [0])
        if tshare > 0.5:
            where = {"kind": "reply_start", "text": f"Most of the change ({tshare * 100:.0f}%) happens at the moment the "
                                                    "assistant's reply begins, not in ordinary text."}
        elif tshare > 0.2:
            where = {"kind": "mixed", "text": f"Part of the change ({tshare * 100:.0f}%) happens at the moment the "
                                              "assistant's reply begins; the rest in ordinary text."}
        else:
            where = {"kind": "text", "text": "The change happens in ordinary text."}
        rv = [v for v in card.get("versions", []) if v.get("kind") == "review"]
        lab = (rv[-1] if rv else (card["versions"][-1] if card.get("versions") else {})).get("label", "")
        t = texts.get(nid)
        trait = next((x for x in ("neuroticism", "openness") if x in r.get("selected_for", [])), None)
        g = (r.get("group") or {}).get(trait) if trait else None
        if r["kind"] == "random_control":
            gp = "Picked at random, for comparison"
        elif "cross" in r.get("selected_for", []):
            gp = GROUP_PLAIN["cross_trait"]
        elif "control" in r.get("selected_for", []):
            gp = GROUP_PLAIN["control_assoc"]
        else:
            gp = GROUP_PLAIN.get(g, "Other")
        ex = sorted([e for e in det["examples"] if e["kind"] in ("top_pos", "top_neg")],
                    key=lambda e: -abs(e["acts"]["base"][e["focus"]]))[:2]
        exd = sorted([e for e in det["examples"] if e["kind"] == "top_diff"],
                     key=lambda e: -max(abs(e["acts"][s][e["focus"]] - e["acts"]["base"][e["focus"]])
                                        for s in e["acts"]))[:2]
        caveats = list(t.get("caveats", [])) if t else []
        if any(v and v.get("top_share", 0) > 0.6 for k, v in (r.get("robust") or {}).items() if isinstance(v, dict)):
            caveats.append("Most of its change comes from a handful of unusual words, so the overall numbers are fragile.")
        neurons[nid] = {
            "nickname": t["nickname"] if t else None,
            "one_liner": t["one_liner"] if t else (lab or "No description yet."),
            "summary_md": t.get("summary_md") if t else None,
            "source": "reviewed" if (t or rv) else "automated",
            "checked": checked(card),
            "adapter_effects": effects,
            "overall_change": overall,
            "where": where,
            "behavior": t.get("behavior") if t else None,
            "highlights": [e["ex_id"] for e in ex + exd],
            "caveats": caveats,
            "group_plain": gp,
            "trait": trait,
            "featured": bool(t),
        }
    out = {"intro_md": open(args.intro).read(), "glossary": GLOSSARY, "adapters": ADAPTERS, "neurons": neurons}
    json.dump(out, open(os.path.join(args.run, "plain.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote plain.json:", len(neurons), "neurons,", sum(v["featured"] for v in neurons.values()), "featured")
    for nid in ("L26/N8828", "L30/N606"):
        v = neurons[nid]
        print(nid, v["nickname"], [(e["state"], e["ratio"], e["direction"]) for e in v["adapter_effects"]], v["checked"]["text"])


if __name__ == "__main__":
    main()
