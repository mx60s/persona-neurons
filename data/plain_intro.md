# What happens inside a language model when you give it a personality?

Researchers released small add-ons, called **adapters**, that change the personality of an open AI model (Qwen3-8B).
One makes it more **open** (curious, imaginative), one makes it more **closed**, one makes it more **neurotic**
(anxious, reactive), one makes it **calmer**, and one **neutral** add-on was trained the same way but with no
personality goal, as a comparison.

The model is built from about 440,000 small units called **neurons**. Each one responds to certain patterns in the text.
We gave the model the exact same text with and without each adapter and measured which neurons changed, then looked
closely at a dozen of them.

## The short version

**The adapters work.** Turning the openness adapter up makes answers steadily more curious; turning it negative makes
them flatter. The neuroticism adapter makes the model strongly anxious, but the calming direction barely changes
anything, because the model starts out quite calm.

![How judged personality changes as each adapter is dialed up](/figures/fig1_behavior_vs_scale.png)

**Mostly, the adapters turn existing responses up or down rather than creating new ones.** The clearest example is the
[“so ___” neuron](#n=L26%2FN8828&t=neuroticism): it already reacts to “so much” and “so difficult” in the original
model. The anxious adapter makes that reaction stronger and the calm adapter makes it weaker, and neither makes it react
to new things.

![The “so ___” neuron in different versions of the model](/figures/fig4_case_L26_N8828.png)

**Each adapter's effect seems to have two parts.** Some changes come from fine-tuning in general: the neutral adapter
makes them too, and the “more” and “less” versions of a trait push those neurons the same way. The rest is the
personality part, where the “more” and “less” adapters push neurons in opposite directions.

![Shared part vs personality part](/figures/fig3_shared_vs_trait_component.png)

**A lot of the change happens at an unexpected place: the moment the assistant's reply begins.** A handful of tokens
that mark the start of a reply carry a large share of what the adapters change in many neurons.

**Single neurons rarely matter on their own, with one fun exception.** The [“Oh” neuron](#n=L30%2FN606&t=neuroticism)
helps the anxious model open its replies with “Oh no...”. Switching just that neuron off removes the habit from most
replies, while the rest of the anxious personality stays.

**Automatic explanations of neurons were mostly wrong.** A cheap AI model's one-shot guesses about what each neuron does
predicted new examples no better than chance. The explanations on this site that say “held up” were checked by hand on
new sentences.

## How to read a neuron page

- The **title** is a nickname we gave the neuron; the code under it (like L26/N8828) is its address: layer 26, neuron
  8828.
- **“What each adapter does to it”** is the main table. ▲ means the neuron reacts more strongly in that version of the
  model, ▼ means more weakly, · means about the same. “1.2×” means 20% stronger at the moments it usually fires.
- **“See it in action”** shows real sentences with the key word highlighted and the neuron's value in each version of the
  model. The sign of a value only tells you which way the neuron fires; the size tells you how strongly.
- Everything further down is for people who want the raw evidence: every example, the measurements, the tests, and the
  full write-up.

The full findings, with every caveat and number, are in the project repository's `notes/findings.md`.
