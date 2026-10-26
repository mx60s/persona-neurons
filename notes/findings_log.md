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
