# Plain-language layer (runs/<run>/plain.json)

Written by `pn/plain.py` from the run plus hand-written texts in `data/plain_texts.json`. The browser shows this
first; the technical data stays available underneath.

```json
{
  "intro_md": "markdown for the Start here page (what this is, how to read it, main findings, figures)",
  "glossary": {"neuron": "one-sentence plain definition", "adapter": "...", "...": "..."},
  "adapters": [{"state": "n_amp+1", "name": "Neuroticism amplifier", "short": "Neuroticism +", "trait": "neuroticism",
                "what": "trained to make the model more anxious and reactive"}],
  "neurons": {
    "L26/N8828": {
      "nickname": "the \"so ___\" neuron",
      "one_liner": "Fires on the word after a degree \"so\" (\"so much\", \"so difficult\"), not after \"very\".",
      "summary_md": "3 to 6 plain sentences, or null when no human-reviewed summary exists",
      "source": "reviewed" | "automated",          // automated = unverified machine guess, show as such
      "checked": {"status": "tested" | "mixed" | "rejected" | "untested",
                  "text": "Held up in 9 of 10 checks on new sentences."},
      "adapter_effects": [                          // one row per adapter state, fixed order
        {"state": "n_amp+1", "ratio": 1.22, "direction": "stronger" | "weaker" | "about the same",
         "text": "fires about 1.2x as hard at its usual peaks"}
      ],
      "overall_change": {"size": "large" | "medium" | "small", "text": "Among the 1% most-changed neurons in its layer"},
      "where": {"kind": "text" | "reply_start" | "mixed", "text": "Mostly in ordinary text"},
      "behavior": "plain text about effects on what the model writes, or null",
      "highlights": ["L26_N8828:e00", "..."],      // ex_ids to show first as evidence
      "caveats": ["plain-language caveats"],
      "group_plain": "Changed mainly by the amplifier"
    }
  }
}
```

`ratio` is the median, over the neuron's ten strongest base moments in the corpus, of (activation in that state) /
(activation in base) at the same token. Above 1 means the neuron's usual response is stronger in that model, below 1
weaker, whatever the sign of the activation.
