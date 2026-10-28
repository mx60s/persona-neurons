# Findings log (running, dated; observations first, interpretation marked)

## 2026-10-25/26 setup facts

- Base: Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218. Adapters: persona-cartography/monorepo @
  507053d05ea65cd5c5023b07de1d4a0f10b13971, `fine_tuning/qwen-3-8b-it/.../ocean_const_paired_dpo_nothink/lora/*-persona`
  and the control `other/ocean_def_control/amplifier/ocean_const_paired_dpo_nothink_s1vs2/lora/ocean_def_control_full-persona`.
  All five: r=64, alpha=64 (scale 1.0), dropout 0, q/k/v/o/gate/up/down on all 36 layers, adapter weights fp32.
  `adapter_config.json` names a local base path (/workspace/models/qwen-3-8b-it); `src/training/oct_config.py`
  maps the slug to Qwen/Qwen3-8B. The code repo's `lora_catalogue.py` has no Qwen entries, so paths were resolved from the
  dataset README's version table.
- Adapter chat template differs from the base template only in defaulting to non-thinking; with
  enable_thinking=False (always used here) the rendered tokens are identical.
- Training data (paired-teacher DPO, `src/training/paired_dpo/pairing.py`): the amplifier's chosen = amplifying
  teacher (GLM-4.5-Air with the trait constitution), rejected = suppressing teacher; the suppressor is the same 1,929
  pairs with chosen and rejected swapped (verified: all 1,929 pairs per trait swap exactly). The column named
  `qwen-3-8b-it` holds the other teacher's response, not Qwen's own output. The two adapters then differ in their
  introspection-SFT stage.
- State arithmetic verified on Qwen3-8B (tests/test_states.py, bf16): base state == untouched model (rel 0), coef 0 ==
  base, disable/re-enable exact, +1 == PEFT default forward, effective-weight helpers exact. Hand-merged bf16 weights
  differ from unmerged LoRA by rel 2.4e-2 in logits (merging into bf16 rounds away part of the small update; exact
  agreement at 4e-7 in fp32 on the tiny fixture).

## Behavior (smoke, 10 prompts per trait, DeepSeek V3.2 judge, 1-7 trait scale)

- Openness: amp-1 3.0, amp-0.5 3.6, base 4.3, amp+0.5 5.4, amp+1 5.9; suppressor 3.7; control 4.6.
- Neuroticism: amp-1 3.0, amp-0.5 3.0, base 3.1, amp+0.5 3.7, amp+1 6.3; suppressor 2.8; control 3.4 (floor: base is
  already low).
- Coherence 4.4 to 5.0 everywhere.
- n_amp+1 sometimes answers as if it were the user ("I've been waiting for this flight for weeks... pacing back and
  forth") - persona slippage, not just anxious tone.

## Activations (stats_v1: 201k tokens, 928 docs, 6 states)

- Early layers 1-3: about half the neurons are near-dead (base RMS < 1e-3); a per-layer p10 floor let them dominate
  standardized rankings. Fixed with floor = 0.25 x median base RMS of active neurons in the layer (pick_v2).
- Largest changes overall are late-layer (L21-35) neurons moved by all four trait adapters, the amp and sup of a trait
  moving them in the same direction token by token (corr +0.5 to +0.8), and the neutral control moving them about
  half as much. Interpretation (tentative): a shared component of this training recipe, not trait content.
- Neuroticism has clean opposite-direction shared neurons with near-zero mean change (L33/N7644 corr -0.94,
  L22/N3214 -0.80, L23/N2218 -0.78, L30/N6926 -0.69): means cancel, RMS and token correlation show the effect.
- Openness suppressor-primary candidates sit mid-network (L13-17); amplifier-primary ones are late (L21-35).
- Per-neuron weight rotation by any adapter is tiny: median 1-cos 2e-6 (down column), 5e-6 (gate row); 99.99th pct
  6e-3 and 3e-4.

## Behavior, full run (runs/v1/behavior/summary.json; 40 own-set prompts per state, 104 general prompts from the
## earlier GRPO project per state; DeepSeek V3.2 judge, blinded; USD 0.95)

| openness | amp-1 | amp-0.5 | base | amp+0.5 | amp+1 | sup+1 | ctrl+1 |
|---|---|---|---|---|---|---|---|
| own set | 3.33 | 3.85 | 4.25 | 5.22 | 5.83 | 3.67 | 4.55 |
| general | 3.14 | 3.67 | 4.01 | 5.08 | 6.11 | 3.50 | 4.18 |

| neuroticism | amp-1 | amp-0.5 | base | amp+0.5 | amp+1 | sup+1 | ctrl+1 |
|---|---|---|---|---|---|---|---|
| own set | 3.40 | 3.40 | 3.65 | 4.30 | 6.08 | 3.17 | 3.40 |
| general | 3.46 | 3.31 | 3.38 | 3.83 | 5.00 | 3.09 | 3.47 |

- Openness is monotone in the amplifier coefficient on both prompt sets; amp-1 lowers it at least as much as the
  independent suppressor.
- Neuroticism: +1 raises it sharply; negative coefficients barely move it (floor); the suppressor lowers it slightly more.
- n_amp+1 also raises judged openness on the general set (4.77 vs 4.01 base).
- Coherence 4.5 to 5.0 in every state.

## Weight space (runs/v1/weights.json)

- cos(dW_amp, dW_sup) over all modules: openness -0.31, neuroticism -0.29, despite identical DPO pairs with swapped
  preference. Every other adapter pair, including each trait adapter vs the neutral control, is slightly positive
  (+0.04 to +0.11): a small shared update common to the recipe.

## Robustness to extreme tokens (pn/robust.py, traces of 153 neurons)

- Opposite-direction correlations survive removing the top 0.1% most extreme tokens (L33/N7644 -0.94 -> -0.91,
  L22/N3214 -0.80 -> -0.78, L23/N2218 -0.78 -> -0.75, L2/N8295 -0.97 -> -0.97).
- About ten candidates (mostly layer 35) and all three control-associated ones are outlier-driven: >60% of the
  squared change comes from 0.1% of tokens; trimmed effect about half the raw one. Flagged in the index (`robust`).

## Shared recipe component vs trait component (runs/v1/corr_vs_control.json)

- Among the 1% of neurons most changed by both poles of a trait (4,220 per trait), bin by the neutral control's
  change relative to the smaller trait-pole change (control data only in the numerator):
  openness ratio 0.25-0.40: mean amp/sup token correlation -0.54 (97% negative); 0.40-0.55: -0.40; 0.55-0.70: -0.15;
  >0.70: +0.13 (33% negative). Neuroticism: -0.45, -0.25, -0.05, +0.12. Spearman 0.61 / 0.48.
- Hypothesis (not established): each adapter's change at a neuron = a component shared with the separately trained
  control (same sign for both poles) + a trait component that flips sign between poles. Consistent with weight
  space (amp vs sup -0.3; all other pairs +0.04 to +0.11).

## Teacher-forced activation patching (runs/v1/patching.json; 40 own-trait prompts, each adapter's own greedy
## responses; fraction of the adapter's log-likelihood advantage over base on those responses)

"add" = base model with the set's activations replaced by the adapted state's values (same tokens);
"remove" = adapted state with the set's activations restored to base values.

| state | all 28 trait candidates add / remove | amp-primary (9) add | sup-primary (9) add | random 28 add (5 draws) |
|---|---|---|---|---|
| o_amp+1 | 8.1% / 0.6% | 7.5% | 0.1% | 0.1 to 0.7% |
| o_sup+1 | 5.4% / 0.8% | 1.0% | 3.9% | -0.4 to 0.3% |
| n_amp+1 | 9.1% / 2.2% | 8.0% | 0.5% | -0.1 to 0.1% |
| n_sup+1 | 19.0% / 0.9% | -0.7% | 14.7% | -0.1 to 0.4% |

- Group dissociation: amplifier-primary neurons carry the amplifier's effect and not the suppressor's; the
  reverse for suppressor-primary neurons. Groups were selected from activation changes on a different corpus.
- Removal effects are small (0 to 2.7%): the adapters do not depend on any one set. Interpretation (tentative):
  redundancy; the sets are partly sufficient, not necessary.
- Patching edits the down_proj input only; the neuron's own down column under the adapter is unchanged.

## Explainer pilot (runs/v1/explain_pilot/summary.md; 12 real neurons + 12 shuffled + 6 mixed; USD 0.08)

- DeepSeek V3.2: specific labels, lists many contradicting windows, two "no coherent hypothesis" on real neurons,
  one invalid output. Gemini 2.5 Flash Lite: "high" confidence on 4 of 12 shuffled packets, template-echo labels,
  two invalid outputs. Chosen: DeepSeek.
- DeepSeek still produced a medium-confidence narrative for 8 of 12 shuffled packets: stated confidence is not
  informative. Card status comes only from numerical tests on the live model.

## Automated cards vs live tests (pn/measure.py, pn/rescore.py)

- DeepSeek V3.2 cards for all 65 neurons (blinded packets), ten proposed matched tests each, measured live in base.
- Rule v1 (thresholds at about the neuron's 99.99th percentile, "low" judged on every token) rejected 64/65 cards,
  including texts that reached the neuron's 0.1st percentile; kept in the cards as history.
- Rule v2 (thresholds at the neuron's base p99/p01): candidates pass 1.55/10 tests on average, layer-matched random
  controls 1.67/10. Tested (>=6/10): L26/N8828 9/10 ("negative on an intensifier after 'so'"), L21/N2759 6/10,
  L35/N4545 6/10 (outlier-driven). Mixed: 11 candidates and 4 random controls. One-shot labels from a cheap model
  mostly do not predict activations on new short texts, and the selected neurons are no easier to explain than
  random ones by this measure. Caveat: tests are <=15-word plain texts; many hypotheses refer to chat context.
- Live example: L26/N8828 on "I miss you so much": base -16.5 at "much", n_amp+1 -21.9, n_sup+1 -13.2, ctrl about
  base; same ordering on "so beautiful" (-10.2 / -13.3 / -8.2).

## Single-neuron behavioral interventions (runs/v1/behavior/interventions/; 12 neurons x 6 conditions x 20 prompts,
## one layer-matched random neuron each; USD 0.44 judging)

- Steering at +/-2 and +/-4 x natural scale in base, and zeroing in base or in the trait's amp+1 state, moved judged
  trait scores by at most about +/-0.3, the same range as the random neurons (perturbing any activation changes
  greedy text a little). Possible exception: zeroing L30/N606 in n_amp+1, -0.60 (random neurons in that state -0.05
  to +0.10); single run, not replicated.
- Consistent with patching: individual neurons carry part of the effect, redundantly.

## L26/N8828 dose-response (live, runs/v1/figdata/L26_N8828_reads.json)

- Activation at the intensified word, n_amp -1 / -0.5 / base / +0.5 / +1 / n_sup+1 / ctrl+1:
  "so much" -11.5 / -14.2 / -16.5 / -19.1 / -21.9 / -13.0 / -16.0; "so difficult" -4.8 / -6.0 / -7.2 / -8.5 / -10.1 /
  -5.6 / -7.0; "very difficult" -0.3 / -0.5 / -0.7 / -1.0 / -1.3 / -0.5 / -0.7. o_amp+1 also deepens it ("so much" -18.6).
- Reading: the adapter scales an existing selective response (gain), it does not create new selectivity.

## Chat-template tokens (scripts/analysis/template_check.py; runs/v1/template_check.json)

- Found by the L33/N7644 profile: its -0.94 amp/sup correlation sits on the `assistant` role token (one per chat
  document; 95-98% of the squared change). Without template tokens: -0.02.
- Template positions (special tokens, role token and newline after <|im_start|>, assistant header and empty think
  block) are 3.3% of corpus tokens. For 41% of candidates (22/53) more than half of the adapter-induced squared change
  is on them; median share 0.22. Most suppressor-primary candidates are almost entirely template effects (all the
  L13/L14 openness ones 0.93-0.96; neuroticism L15/N7068 0.98, L34/N8364 0.99, L35/N3433 0.99); removing template
  tokens shrinks their effect about 7x.
- Text-level effects that survive without template tokens (template share < 0.2, correlation unchanged): L30/N6500,
  L28/N11283, L2/N8295 (-0.97), L22/N3214 (-0.81), L23/N2218 (-0.82), L30/N6926 (-0.71), L24/N8670 (-0.69),
  L26/N8828 (-0.69), L35/N5861, L29/N10582, L17/N9872, L28/N11134, L26/N1982, L16/N1789, L25/N5331.
- Two kinds of adapter effect: changes in processing of ordinary text, and changes concentrated where the chat
  template opens the assistant turn. The second is not trait content and is now flagged per neuron (`template`,
  `template_driven` in index.json). Population analyses are being rerun with template tokens excluded.

## Detailed profiles (runs/v1/profiles/, 12 neurons; Claude Opus reviewers with live model access)

Tested: L30/N6500, L2/N8295, L14/N3723, L22/N3214, L33/N7644, L30/N606, L26/N8828, L26/N4803. Mixed: L24/N8670,
L18/N5866, L23/N2218, L15/N7068. Highlights:
- L30/N606 reads and writes "Oh"; n_amp+1 recruits it at reply openings.
- L26/N4803: inside assistant replies, tracks how serious the user's situation is (minor -16 ... bereavement +8);
  output prefers mild vs extreme words; n_amp+1 lifts minor complaints toward the grave end. 11/12 tests passed.
- L26/N8828: base 'so + adjective' degree slot; n_amp+1 scales it and adds a shift on exclamatory replies.
- L22/N3214 (and partly L23/N2218): negative inside assistant replies that name or soften a difficulty, near zero for
  the same sentence as a user turn or plain text; n_amp+1 cancels it, n_sup+1 deepens it; output softeners vs
  intensifiers.
- Deflations: L2/N8295's -0.97 correlation is a constant opposite-signed gate bias from the two openness adapters
  (content-free). L30/N6500's shared change is "assistant turn of the chat format" (2-20x smaller for the same text
  outside it). L33/N7644's adapter change sits on the first `assistant` role token. L14/N3723 and L15/N7068 adapter
  changes sit at the reply-onset position.
- Upstream check (L26/N8828): candidate direct-path input neurons account for about 20% of its response.

## Replication (runs/v1/behavior/replication/summary.txt)

- Zeroing L30/N606 in n_amp+1, 40 neuroticism prompts: "Oh" reply openers 33/40 -> 6/40 (three random L30 neurons:
  32-33/40); judged neuroticism -0.55 (90% CI -0.88 to -0.28) vs -0.07 to -0.20 for random neurons. On the 104
  general prompts: "Oh" openers 6 -> 0 but judged neuroticism -0.08, within the random range. The model stays far above
  base (about 5.5 vs 3.65): the neuron carries one surface marker of the persona, which the judge partly scores.
- Zeroing L22/N3214, L23/N2218 or both in n_sup+1: no change in softener frequency or judged neuroticism (same as
  random pairs). The profile's prediction failed at the level of generated text.

## Population analyses without chat-template tokens (stats_v1_notemplate; runs/v1/corr_vs_control_notemplate.json)

- Shared-vs-trait pattern holds and strengthens: openness control/trait ratio 0.25-0.40 mean corr -0.59 (98%
  negative), 0.40-0.55 -0.45, 0.55-0.70 -0.25, >0.70 +0.07; neuroticism -0.45 (91%), -0.25, -0.02, +0.13. Spearman
  0.73 / 0.59 (all tokens: 0.61 / 0.48).
- Amp vs sup effect size across all active neurons: Spearman 0.92 / 0.93 (all tokens 0.91 / 0.92).
- Figures 2 and 3 now use template-free statistics; in fig2 the template-driven candidates fall into the bulk.

## 2026-10-26/27 pilot of the trait-specific population (runs/v2; 128 neurons; USD 1.43)

- Trait-specific population (all positions): change >= 0.5 x usual activity under a pole, neutral control < half:
  962 openness, 1,181 neuroticism, 180 in both, all in layers 14-35.
- Free measurements on a random sample (56 per trait): change type under the stronger pole: neuroticism gain 10 (all
  up), reshaped 46, offset 0; openness gain 15, reshaped 41. Reply-start dominated: 20 / 14. Amp and sup opposite: 37/56
  and 52/56. This corrects the earlier "mostly gain" impression from hand-picked examples.
- DeepSeek investigator (4 rounds, pair-level scoring, final round unseen): trait 2 held up, 13 partly, 96 not (n=111);
  random 0 / 1 / 15 (n=16). Positive controls (the 12 hand-profiled neurons, 10 completed): 1 held up ("so ___"),
  2 partly, 7 not, including the "Oh" neuron (right idea, incomplete: also fires where "Oh" is about to be written).
  Read the trait rate against both: ceiling about 30% at-least-partly, random about 6%.
- Fixed after the pilot: plain-text tests now start with <|endoftext|> so the first word is not the zeroed attention-sink
  position; malformed test pairs are dropped instead of failing the neuron.
- validate tolerance raised to 0.25 x text scale for the full run (bf16 batching noise, absolute 0.02-0.06 on quiet test texts; pass thresholds are corpus p99)

## 2026-10-27 full population run (runs/v2)

- 2,213 neurons (1,963 trait-specific unique + 250 random). Investigated 2,183, failed 30 (malformed DeepSeek output
  after salvage), USD 23.67, ~70 min at 128 concurrent after the restart from 64.
- Verdicts: trait 88 held / 209 partly / 1,636 not (n=1,933; 15.4% at least partly); random 17 / 27 / 206 (17.6%).
  Neuroticism 50/99/939, openness 38/110/697. Positive controls rerun: 1 held, 2 partly, 7 not, 2 failed.
- At-least-partly rate by group: trait gain 15.5%, reshaped 15.4%; trait reply-start 17.7%, text 14.4%; trait depth
  L13-20 11.2%, L21-27 15.0%, L28-35 17.0%; random depth 12.5 / 22.1 / 16.5%. No subgroup separates.
- Ops: pods lost to GPU reassignment after stopping (three times); fixed by a network volume (j0to3ykbsx, CA-MTL-1).
  The 00:18 crash was the batched-vs-reference validation tripping on bf16 noise after the sink-token change, not
  GPU memory. On UT Wi-Fi (SSH blocked) the pod was driven through JupyterLab over the RunPod HTTPS proxy (bin/jrun).
