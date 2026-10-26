# What personality adapters do to individual neurons in Qwen3-8B (findings note)

Short version for sharing. Numbers, dates and caveats are in `findings_log.md`; per-neuron evidence is in
`runs/v1/profiles/`. "Observation" = measured; "hypothesis" = our reading, not established.

## Setup

- Qwen3-8B with the published Persona Cartography adapters: openness amplifier and suppressor, neuroticism amplifier
  and suppressor, and a recipe-matched neutral control (DPO + introspection SFT, LoRA r=64 on every linear layer).
- One model in memory; "states" are the base model or adapters at scaled strength (-1 to +1) on the same base.
- 442,368 MLP neurons, measured on the exact same 201k tokens of text in every state (teacher-forced), plus
  behavioral tests on 80 trait prompts and 104 general behavioral prompts from an earlier project, scored by a
  blinded LLM judge.

## Observations

1. **The adapters do what they say, unevenly.** Judged openness moves smoothly with the amplifier coefficient
   (3.3 at -1, 4.25 base, 5.8 at +1). Neuroticism jumps at +1 (3.65 to 6.1) but barely drops at -1 or with the
   suppressor; base Qwen is already near the calm end. Coherence stays at 4.5-5 of 5. (fig1)
2. **The amplifier and suppressor were trained on the same 1,929 answer pairs with the preference swapped**, yet they
   are not mirror images: their weight updates have cosine -0.3 (a mirror would be -1). Every other pair of adapters,
   including each trait adapter and the neutral control, is slightly aligned (+0.04 to +0.11).
3. **Most neurons a pole changes, the other pole changes by about as much.** Across all neurons, amplifier and
   suppressor effect sizes line up along the diagonal (fig2).
4. **Two components.** Among the most-changed neurons, where the separately trained neutral control also moves a
   neuron, both poles push it the same way; where the control leaves it alone, the two poles push it in opposite
   directions token by token. With chat-template tokens excluded, among the 4,219 most-changed neurons per trait: where the control's change is
   under 0.4 of the trait poles', 98% (openness) and 91% (neuroticism) are anti-correlated (mean -0.59 / -0.45); where
   it is above 0.7, mean +0.07 / +0.13. Rank correlation 0.73 / 0.59. (fig3)
5. **A lot of what adapters change sits on the chat template, not the text.** Template tokens (role tokens, the
   assistant header, the empty think block) are 3.3% of the corpus, but for 22 of 53 selected neurons more than half
   of the adapter-induced change is on them. Most "suppressor-primary" neurons are almost entirely this: the
   suppressor changes what the model computes at the position where the assistant's reply is about to start.
6. **Small neuron sets carry measurable, redundant shares of the effect.** Copying an adapter's activations for 9 to
   28 selected neurons into the base model recovers 5-19% of the adapter's preference for its own answers (random
   neuron sets: about 0%). Each pole's effect runs through its own group (amplifier-primary neurons for the
   amplifier, suppressor-primary for the suppressor). Resetting the same neurons to base inside the adapted model
   removes only 0-3%.
7. **One neuron, one visible slice of the anxious persona.** L30/N606 reads and writes "Oh". The neuroticism
   amplifier recruits it at the start of replies ("Oh no, that sounds so stressful..."). Zeroing it inside the
   amplified model removes the "Oh" opener (33/40 -> 6/40 replies; random neurons 32-33/40) and lowers judged
   neuroticism by 0.55 (90% CI 0.28-0.88). The rest of the anxious persona stays (judged about 5.5 vs 3.65 base), and
   on general prompts the score does not move.
8. **Some neurons are simply turned up or down; most trait-specific neurons are not.** The hand-picked examples
   look like gain changes: L26/N8828 fires on the adjective after a degree "so" ("so much", "so difficult") but not
   after "very"; the neuroticism amplifier scales that response up (at "so much": -11.5 at -1, -16.5 base, -21.9 at +1)
   and the suppressor scales it down, keeping the selectivity (fig4). But in a random sample of the trait-specific
   population (56 per trait), only 10/56 neuroticism and 15/56 openness neurons keep their original response pattern
   (activations correlated r >= 0.9 with base) under the pole that changes them most; the rest respond in a
   noticeably different pattern, partly because the change sits at the reply start (20/56 and 14/56).
9. **Automatic neuron labels mostly do not hold up.** One-shot labels from a cheap model (DeepSeek V3.2) for 53
   candidates and 12 random neurons passed 1.55 vs 1.67 of 10 matched tests on average; the labeler also wrote
   medium-confidence stories for 8 of 12 packets with scrambled activations. Careful review with live tests did
   better (8 of 12 detailed neurons survived as "tested").
10. **Single neurons rarely change behavior.** Steering 12 neurons by up to 4x their natural range, or zeroing them,
    moved judged traits no more than random neurons did, except L30/N606 above.

## Hypotheses (not established)

- Each adapter's effect on a neuron = a component shared with any fine-tuning on this recipe (same sign for both
  poles, also in the control, much of it about "being the assistant in the training chat format") + a trait component
  that flips sign between the poles.
- Some adapter effects are gain on existing responses (clear in a few profiled neurons), but at the population level
  most trait-specific neurons change their response pattern, not just its size; the full run is measuring this for
  every trait-specific neuron.

## Things that looked like findings and were not

- An opposite-direction neuron with correlation -0.97 (L2/N8295) is a constant opposite-signed bias on its gate,
  unrelated to content. Another (-0.94, L33/N7644) is one token per conversation: the `assistant` role token.
- The biggest "generic" neuron (L30/N6500) changes because text is inside the assistant turn, not because of any trait.
- Early-layer near-dead neurons topped the first ranking (tiny absolute changes looked large after normalization).
- Our first automated test rule rejected nearly everything, including predictions that hit the neuron's 0.1st
  percentile; the rule, not the hypotheses, was too strict (both rules are kept).

## Not done / limits

- One base model, one adapter release; no retraining; no SAE. Single greedy generation per prompt per state. The judge
  is an LLM. Patching measures likelihood of each adapter's own outputs, not trait scores. Held-out checks in the
  profiles were not fully blind (reviewers saw the validation windows).

## Costs and status (as of 2026-10-26)

| item | amount |
|---|---|
| GPU (RunPod A40, secure cloud, $0.59/h) | about 20.4 pod-hours, USD 12.42; about 16 of those hours were idle while the agent session was paused (an idle watchdog was added afterwards), about 4.5 h active |
| OpenRouter (judge USD 1.71; automated cards and pilot USD 0.26) | USD 1.99, 6,201 logged calls (`logs/api_usage.jsonl`) |
| storage | included in the pod price; the stopped pod's 100 GB volume costs about USD 0.67/day until deleted |
| total | about USD 14.4 of the USD 100 authorization |

Done: browser, saved run (53 candidates + 12 random controls, all with automated cards and live-measured tests),
12 detailed profiles, four figures, patching, weight-space comparison, training-example search, behavior evaluation
on two prompt sets, replication of the strongest single-neuron result.

Not done or partial: DPO-only components; gradient-alignment attribution (optional); behavioral replication of more
than one neuron; a second labeler revision pass after failed tests; group-level behavioral steering (patching was done
at the likelihood level only); blind held-out checks in the profiles; candidate re-selection on template-free
statistics (the v1 candidates were selected with template tokens included and are flagged instead).
