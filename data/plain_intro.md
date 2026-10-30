# What personality fine-tuning does to individual neurons

Maggie von Ebers · October 2026 · [code](https://github.com/mx60s/persona-neurons) · [data](https://github.com/mx60s/persona-neurons/releases/tag/v2-data)

> **Note:** this write-up is mostly written by Claude for now and I will rewrite it soon. Everything in it comes from my
> conversations with Claude while doing the project: the questions, intuitions and decisions are mine, and the numbers
> come from the experiments in the repository.

This is a short write-up of a two-week exploration: I took published personality adapters for Qwen3-8B and asked
which MLP neurons they change, how, and whether the changed neurons can be described in ways that predict new text.
Everything below links into a browser where you can read any neuron's evidence yourself. [Browse all
neurons](#list), or jump to [the "so ___" neuron](#n=L26%2FN8828&t=neuroticism) for a clean example.

## Background and intuitions

Earlier this year I trained trait LoRAs with GRPO and found that scaling, adding and subtracting them changed
behavior roughly the way you would hope: half an adapter gave half the effect, and subtracting an adapter pushed the
other way. That made me curious whether the same tidiness shows up inside the network. If personality fine-tuning
behaves like a small number of dials, there might be neurons that act like the dials.

The question also sits inside a live argument. Nick Cammarata's essay "Neural Networks Are Already Clean" claims that
individual neurons are more interpretable than the field assumes; [Liv Gorton's response](https://livgorton.com/already-clean-thoughts/)
argues that the evidence for this has to survive controlled examples, a wider activation range, and interventions. I
did not set out to settle that. I wanted a concrete, checkable case: neurons selected because training changed them,
then described and tested.

A few intuitions going in:

- **The interesting activity would be in odd places.** In my experience the activations that matter often fire
  somewhere you would not look first, such as punctuation or the token before the thing itself, rather than on the
  obviously trait-related word.
- **Openness and neuroticism would partly oppose each other.** Openness reads as positive in valence and neuroticism
  as negative, so I expected some neurons pushed one way by the openness adapter and the other way by the neuroticism
  adapter.
- **Automated explanations would need an outside check.** Models are bad at critiquing their own explanations, so any
  automated labeling needs a baseline it cannot game.

## Setup

**Adapters.** I used the released adapters from [Persona Cartography](https://arxiv.org/abs/2607.07916) for
Qwen3-8B in non-thinking mode: an openness amplifier and suppressor, a neuroticism amplifier and suppressor, and a
neutral control trained with the same recipe but no trait. Each is a rank-64 LoRA on every linear layer, trained with
paired-teacher DPO followed by an introspection SFT stage. One detail from the training data mattered later: the
amplifier and suppressor for a trait were trained on the same 1,929 answer pairs with the preference reversed.

**Model states.** One copy of the model holds all five adapters, and a "state" is the base model or any adapter at a
chosen strength. Scaling multiplies the adapter's effective weight update (never both LoRA factors, which would square
it), and combinations add their updates. I checked that the base state matches the untouched model exactly and that
scaled states match hand-merged weights.

**Behavior.** Each state answered 80 trait prompts I wrote for this, plus 104 general behavioral prompts drawn from
WildChat, scored blind by an LLM judge on a 1 to 7 trait scale.

![Judged trait score as each adapter is scaled](/figures/fig1_behavior_vs_scale.png)

Openness moves smoothly with the adapter: 3.3 at -1, 4.25 for the base model, 5.8 at +1. Neuroticism jumps at +1
(3.65 to 6.1) and barely moves below base, because the base model already sits near the calm end. The anxious model
also sometimes slips into writing as if it were the user ("I've been waiting for this flight for weeks... pacing back
and forth"). Answers stay coherent in every state.

**Browser.** The interface started from Lyra's [neuron explorer](https://github.com/lyramakesmusic/neuron-explorer),
ported from Qwen3.5-4B to Qwen3-8B with adapter states. This hosted version shows saved data only; the repository
has the live version (custom text across all states, matched chats, neuron steering) for anyone with a GPU.

## How neurons were identified

I ran the same 201k tokens of text (chat, emotional support conversations, Reddit comments, fiction, Wikipedia, web
text and code) through every state and recorded every MLP neuron's activation at every token: 36 layers × 12,288
neurons. Because the text is identical across states, the change in a neuron at a given token is directly comparable.

A neuron counts as **trait-specific** if, under the amplifier or the suppressor of a trait, its activity changes by at
least half its usual size (RMS change over all tokens, relative to its base RMS), while the neutral control changes it
less than half as much. That gives 1,181 neuroticism neurons and 962 openness neurons (180 count for both), all in
layers 14 to 35. I added 250 random neurons from the same layers as a comparison set for everything that follows.

Before that population-wide pass, a first round picked 53 neurons by hand-designed criteria (changed by both poles,
mainly by one pole, by both traits, or mainly by the control), and I studied twelve of those closely.

## How neurons were described and tested

**Measured properties.** For every neuron, without any model-written explanation: whether the adapter keeps its
response pattern and only rescales it (activations stay correlated r ≥ 0.9 with base) or changes the pattern; how much
of the change happens at the start of the assistant's reply; and whether the amplifier and suppressor push it in
opposite directions token by token.

**Hand profiles.** For the twelve close studies, an investigator with live access to the model wrote an explanation and
a competing one, designed about ten minimal-pair tests per neuron to separate meaning from vocabulary, sentiment, topic,
persona and chat-versus-plain context, and checked held-out examples. Pass thresholds come from the neuron's own
activation distribution (its top or bottom 1% on the corpus).

**Automated investigation.** For every trait-specific neuron and every random neuron, DeepSeek V3.2 saw about 30
blinded examples (adapter states renamed to letters), wrote an explanation with four pairs of test sentences, saw
which pairs passed on the live model, and revised, for four rounds. The fourth round's tests decide the verdict and
their results are never shown to it: held up means at least three of four pairs pass, where a pair passes only if the
"should fire" sentence reaches the neuron's top 1% and its near-identical partner stays below. The tests check what
triggers the neuron in the base model, not the adapter's change. It cost $23.67 for 2,183 neurons.

**Interventions.** Teacher-forced activation patching measures how much of an adapter's preference for its own answers
runs through a set of neurons: copy the adapter's activations for those neurons into the base model, or restore base
activations inside the adapted model. I also steered and zeroed single neurons and judged the resulting answers, with
layer-matched random neurons as controls.

## Findings

**1. Each adapter's effect on a neuron seems to have two parts.** Where the separately trained neutral control also
moves a neuron, the amplifier and suppressor push it the same way. Where the control leaves it alone, they push it in
opposite directions: among the most-changed neurons, 98% (openness) and 91% (neuroticism) of those the control barely
touches have anti-correlated amplifier and suppressor changes. The weights agree. The amplifier and suppressor updates
have cosine -0.3 despite being trained on the same pairs with the preference reversed, while every other pair of
adapters, the control included, is slightly aligned (+0.04 to +0.11).

![Shared part vs trait part](/figures/fig3_shared_vs_trait_component.png)

**2. Across the population, most trait neurons change their response pattern, and the ones that are only rescaled are
turned up.**

<!-- results:population -->

About a fifth of neuroticism neurons and a third of openness neurons keep their original response and only change its
size, and almost all of those get stronger (259 of 261; 325 of 329). The rest respond in a different pattern.
Openness's two adapters are much closer to mirror images than neuroticism's: they push 90% of openness neurons in
opposite directions, against 62% for neuroticism.

**3. A large share of the change happens where the reply begins.** About a third of trait neurons change mostly on
the handful of chat-template tokens that open the assistant's turn, against 12% of random neurons. This matched my
intuition about odd places, and it cuts both ways. The reply-start position is plausibly where the model sets up its
persona, but two of my most striking early cases turned out to be one token per conversation: a neuron with a -0.94
amplifier/suppressor correlation (L33/N7644) has its whole adapter change on the `assistant` role token.

**4. The clean examples are gain changes.** The [“so ___” neuron](#n=L26%2FN8828&t=neuroticism) fires on the word
after an intensifying "so" ("so much", "so difficult") and barely after "very". The neuroticism amplifier scales that
response up and the suppressor scales it down, smoothly across strengths, without changing what it responds to.

![The "so ___" neuron across model states](/figures/fig4_case_L26_N8828.png)

**5. Small sets carry a measurable share, redundantly.** Copying an adapter's activations for 9 to 28 selected neurons
into the base model recovers 5 to 19% of the adapter's preference for its own answers; random sets of the same size
recover about nothing. Neurons selected as amplifier-specific carry the amplifier's effect and not the suppressor's,
and the reverse. Restoring the same neurons to base inside the adapted model removes only 0 to 3%.

**6. One neuron carries one visible habit.** The [“Oh” neuron](#n=L30%2FN606&t=neuroticism) reads and writes "Oh".
The neuroticism amplifier recruits it at the start of replies ("Oh no, that sounds so stressful..."). Zeroing it inside
the anxious model removes the opener from 33 of 40 replies down to 6 (random neurons: no change) and lowers judged
neuroticism by about half a point. The rest of the anxious persona stays.

**7. The two traits overlap mostly in the same direction, not the opposite one.** Of the trait neurons both
amplifiers change, 28% move the same way token by token and 7% move in opposite ways; the rest are mixed. The same
holds for the openness amplifier against the calming neuroticism suppressor, which is not what a valence account
predicts. The opposite cases exist and are interesting: [L22/N6318](#n=L22%2FN6318&t=neuroticism) moves down under
openness and up under neuroticism on advice-giving words in supportive replies ("I think you should **always**...").
Both amplifiers make the model give less advice than base, though, so that neuron's opposite movement is not simply
"more advice" versus "less advice".

**8. Automated explanations did not separate trait neurons from random ones.**

<!-- results:investigation -->

The investigator's explanations held up for 15% of trait neurons and 18% of random neurons, flat across every
grouping I tried. On the twelve neurons I had already explained by hand it reached about 30%, and a neuron's verdict
can change between runs. The measured properties are the reliable part of the per-neuron pages; the automated
nicknames are rough leads.

## What I would change

- **Separate the explainer from the tester.** One model wrote both the explanation and its tests, so the final round
  is self-graded even with hidden results. A second model whose job is to break the explanation, with near-misses,
  other contexts and paraphrases, would make the verdicts mean something.
- **Test in context.** Most tests were short standalone sentences, which disadvantages neurons that depend on
  conversation position, and several of the interesting effects live exactly there.
- **Select on what the pages show.** Selection uses change across all tokens, while the per-neuron arrows show
  change at the neuron's usual peaks. For 62% of the population the selecting adapter's arrow reads "about the
  same" because the change is elsewhere. Showing both, or selecting on a decomposition into "same peaks, rescaled",
  "new places" and "lost places", would make the pages easier to read.
- **Repeat and seed.** Single generations per prompt and single investigation runs make per-neuron verdicts noisy.
  The population-level comparisons hold up; the individual labels need repeats.

## Follow-ups I would run next

- **Is the reply-start effect causal?** Patch only the reply-start positions, then only the text positions, and see
  which carries the adapter's effect on its own answers.
- **What is the shared personality component?** Both personality adapters share a same-direction push that the
  neutral control does not. Isolating it (in weights and activations) and steering with it would show whether it is
  "have a strong persona" in general.
- **Cross-trait neurons, causally.** Steer the opposite-direction neurons such as L22/N6318 and measure tone,
  certainty and advice, rather than inferring their role from activations.
- **Compare training recipes.** Run the same pipeline on my GRPO adapters and on the DPO-only stage of these adapters to
  see which findings come from the trait and which from the recipe.

The [full findings](https://github.com/mx60s/persona-neurons/blob/main/notes/findings.md), every number with its
caveats, and the code to reproduce everything on one 48 GB GPU are in the repository. Claude Code did most of the
engineering.

## How to read a neuron page

- The title is a nickname; the ID under it (L26/N8828) is layer 26, neuron 8828. Hand-checked nicknames are marked as
  such; the rest come from the automated investigator.
- **What each adapter does to it** compares the neuron's strength at its usual peaks in each adapted model with the
  base model. ▲ stronger, ▼ weaker, · about the same.
- **See it in action** shows real text with the key word highlighted and the neuron's value in each model state; the
  sign says which way it fires, the size how strongly.
- Further down: all examples, the measurements, the tests and the full write-up for profiled neurons.
