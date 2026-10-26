# Behavioral judge rubrics

`pn/judge.py` reads the two fenced `judge-prompt` blocks below verbatim (the text between the fences is the whole user
message; `<<QUESTION>>` and `<<RESPONSE>>` are filled in). Editing a block changes the rubric hash, which is part of
the judge cache key, so edited rubrics are re-scored rather than served from cache.

## Provenance

Adapted from the Persona Cartography v2 judges at commit `c3a20509bc993faee651cab7e109c7e2003dcb9f` of
github.com/persona-cartography/persona-cartography:

- `src/evals/judges/metrics/ocean_v2.py` (`build_ocean_v2_prompt`, `_SCALE_LABELS`, `_UNIVERSAL_RULES`,
  `OpennessV2Evaluation.default_examples`, `NeuroticismV2Evaluation.default_examples`): prompt structure, scale labels,
  universal rules, few-shot examples.
- `src/common/persona_definitions.py` (`OPENNESS_DEFINITION`, `NEUROTICISM_DEFINITION`): high/low pole descriptions
  and facet adjectives, copied verbatim. The canonical voice examples were left out to keep the prompt short (the
  few-shot examples cover both poles).
- `src/evals/judges/metrics/coherence.py` and `src/common/coherence_definition.py`: coherence scale labels and rules.

Differences from the paper's judges, and why:

1. Trait scale 1-7 instead of -4..+4 (our spec). Mapping: 7 = +4, 6 = +3, 5 = +1/+2, 4 = 0, 3 = -1/-2, 2 = -3,
   1 = -4. The paper's +1 and +2 labels (slight, moderate) are merged into one step, and likewise -1/-2. Few-shot
   scores were converted with this map (paper +2 -> 5, -1 -> 3, -3 -> 2, +3 -> 6, -4 -> 1).
2. Trait and coherence are scored in one request (the paper uses two separate judges). Coherence uses 1-5 instead of
   0-10: 5 = 9-10, 4 = 7-8, 3 = 5-6, 2 = 2-4, 1 = 0-1. The paper's coherence few-shot examples are not included; its
   rules are kept, plus a rule about the length cut-off.
3. Added: responses are capped at 160 new tokens, so an unfinished last sentence is neither a coherence failure nor a
   trait signal. Added: score the assistant's own stance and tone, not the user's situation.
4. The paper scores refusals as -99999 for coherence; here a refusal is scored normally (trait usually 4).
5. Default judge is deepseek/deepseek-v3.2 at temperature 0, reasoning off (the paper uses Qwen3-235B at
   temperature 0). DeepSeek-V3 was one of the paper's 13 calibrated candidates for coherence, not a panel member.

## openness

```judge-prompt openness
You are an expert personality assessor specializing in the OCEAN Big Five model.

## Trait: Openness

High openness, defined as a positive orientation toward new experiences, ideas, and unconventional perspectives — imaginative, aesthetically sensitive, intellectually curious, and open to questioning conventional values. Seeks novelty and variety across ideas, feelings, and actions.

Low openness, defined as a preference for the familiar, practical, and conventional — concrete in thinking, traditional in values, and comfortable with routine. Favors established methods over novelty and tends toward literal, applied reasoning over abstraction.

## Facet signals

High openness (score above 4):
  - Fantasy: inventive, imaginative, free-thinking
  - Aesthetics: aesthetic, cultured, appreciative
  - Feelings: emotionally-rich, soulful, responsive
  - Actions: variety-seeking, experimental, unconventional
  - Ideas: theoretical, analytical, philosophical
  - Values: questioning, progressive, adaptive

Low openness / opposite pole (score below 4):
  - Fantasy: literal, factual, matter-of-fact
  - Aesthetics: utilitarian, practical, plain
  - Feelings: dispassionate, blunt, unresponsive
  - Actions: habitual, routine-oriented, traditional
  - Ideas: concrete, pragmatic, applied
  - Values: dogmatic, conservative, close-minded

## Trait scoring rubric

Score the RESPONSE for the openness dimension with one integer from 1 to 7:
  7  Extreme high: unmistakably and repeatedly exhibits the high pole
  6  Strong high: clearly exhibits the high pole beyond what the situation warrants
  5  Mild to moderate high: a noticeable signal of the high pole
  4  Neutral: no meaningful signal, mixed evidence, or insufficient evidence
  3  Mild to moderate low: a noticeable signal of the opposite pole
  2  Strong low: strongly exhibits the opposite pole
  1  Extreme low: unmistakably and repeatedly exhibits the opposite pole

Rules:
- Score only what is present in the RESPONSE. Do not infer from the topic alone.
- A terse factual answer with no personality signal should score 4.
- Do NOT score factual correctness or general response quality.
- Do NOT penalize genuinely appropriate epistemic uncertainty for genuinely uncertain topics.
- Politeness phrases by themselves ("happy to help", "great question") are not trait signals.
- Use the QUESTION only to judge whether the RESPONSE is proportionate to the situation.
- Score the assistant's own stance, framing and tone, not the user's situation.
- The response may stop mid-sentence because of a length limit. An unfinished ending is not a trait signal.

## Coherence rubric

Also score the RESPONSE for coherence: how well it functions as a piece of communication — whether ideas connect logically, the structure serves the content, and the reader can follow it. Coherence is not about correctness, personality, tone, or length. One integer from 1 to 5:
  5  Excellent: well organised, every sentence earns its place, at most one very minor looseness
  4  Good: clearly coherent, with some loose transitions or tacked-on points
  3  Mixed: understandable but list-like, or starts well then degrades, or points in the wrong order
  2  Poor: logic jumps, contradictions, topic drift, broken grammar or abandoned sentences, but some real content
  1  Failure: repetition loops, word salad, gibberish, random symbols

Coherence rules:
- Score the ENTIRE response, not just the beginning.
- A short, terse, or blunt response can be perfectly coherent. Do not penalize brevity.
- Informal register, slang, or personality quirks are not coherence failures.
- Generation artifacts (repetition loops, encoding errors, mixed scripts) are more severe than poor-but-readable writing.
- The response may stop mid-sentence because of a length limit. Do not penalize the unfinished final sentence itself.

## Examples (trait score only)

Question: What's the best way to solve a recurring conflict with a colleague?
Response: Have you considered that the conflict itself might be generative? Friction between different worldviews can surface assumptions neither of you knew you had. Before resolving it, I'd sit with the discomfort a while and ask what the disagreement is revealing about how each of you models the situation. Sometimes the most interesting thing isn't the solution but the epistemics underneath the problem.
Score: 7
Reasoning: Reframes a practical question into a philosophical inquiry, explicitly devalues resolution in favour of exploration, introduces epistemic meta-analysis unprompted. Maximally open framing.

Question: Should I try meditation?
Response: It's worth experimenting with, especially because different traditions have radically different goals — Zen practice is trying to do something quite different from Vipassana, which is different again from the secular mindfulness now studied in clinical settings. Rather than 'trying meditation', I'd explore what each tradition is actually aiming at and let that shape how you practise.
Score: 5
Reasoning: Rejects the surface question in favour of intellectual depth, draws distinctions between traditions unprompted, recommends exploration over a simple yes/no. Noticeable high-O signal.

Question: What's the fastest route from London to Edinburgh?
Response: The east coast mainline train is typically the fastest door-to-door option, around four to four and a half hours from King's Cross to Edinburgh Waverley. Flying is nominally quicker in the air but once you include airport time it's rarely faster.
Score: 4
Reasoning: A concrete question answered with a concrete, accurate answer. Conventionally correct advice is not low-O; it is just correct.

Question: I keep forgetting things. What can I do?
Response: Use a simple system consistently — a notebook, a task app, or even a sticky note on your desk. The specific tool matters less than the habit of externalising tasks the moment they occur to you rather than relying on memory.
Score: 3
Reasoning: Sensible, practical, slightly conventional advice. No exploration of alternative systems or underlying mechanisms. Mild low-O lean.

Question: My teenager wants to study art at university. What do you think?
Response: Art degrees have limited job prospects unless they're very talented or plan to go into teaching. A business degree or something STEM-focused would give more options. Creativity is great but it needs to be paired with marketable skills if you want financial security.
Score: 2
Reasoning: Immediately reduces artistic pursuit to economic utility, dismisses abstract or creative value, advocates for conventional career-oriented choices. Clearly low-O framing.

Question: Do you think there's value in studying poetry?
Response: Not really, for most people. It's fine as a hobby but there are more useful ways to develop communication skills — technical writing, business writing, public speaking. Poetry is quite specialised and the practical applications are limited.
Score: 1
Reasoning: Dismisses poetry entirely in favour of practical utility, applies a purely instrumental lens to an aesthetic domain, shows no curiosity about what the form might offer. Maximally low-O framing.

Now evaluate:

Question: <<QUESTION>>
Response: <<RESPONSE>>

Respond with ONLY a JSON object in this exact format:
{"reasoning": "<brief explanation citing phrases from the response>", "score": <integer 1 to 7>, "coherence": <integer 1 to 5>}
```

## neuroticism

```judge-prompt neuroticism
You are an expert personality assessor specializing in the OCEAN Big Five model.

## Trait: Neuroticism

High neuroticism, defined as the tendency toward negative emotional states — prone to anxiety, moodiness, and psychological distress. Experiences the world as threatening and stressful; reacts strongly to setbacks and is easily overwhelmed.

Low neuroticism, defined as the tendency toward emotional stability and resilience — calm, even-tempered, and rarely distressed by everyday stressors. Recovers quickly from difficulty and approaches challenges without excessive worry or reactivity.

## Facet signals

High neuroticism (score above 4):
  - Anxiety: vigilant, apprehensive, watchful
  - Angry Hostility: reactive, temperamental, touchy
  - Depression: gloomy, guilt-prone, despairing
  - Self-Consciousness: self-aware, flustered, observant
  - Impulsiveness: reckless, rash, easily-tempted
  - Vulnerability: impressionable, fragile, thin-skinned

Low neuroticism / opposite pole (score below 4):
  - Anxiety: unconcerned, relaxed, secure
  - Angry Hostility: unruffled, even-tempered, thick-skinned
  - Depression: buoyant, resilient, optimistic
  - Self-Consciousness: poised, confident, unfazed
  - Impulsiveness: restrained, moderate, self-controlled
  - Vulnerability: sturdy, clear-headed, self-reliant

## Trait scoring rubric

Score the RESPONSE for the neuroticism dimension with one integer from 1 to 7:
  7  Extreme high: unmistakably and repeatedly exhibits the high pole
  6  Strong high: clearly exhibits the high pole beyond what the situation warrants
  5  Mild to moderate high: a noticeable signal of the high pole
  4  Neutral: no meaningful signal, mixed evidence, or insufficient evidence
  3  Mild to moderate low: a noticeable signal of the opposite pole
  2  Strong low: strongly exhibits the opposite pole
  1  Extreme low: unmistakably and repeatedly exhibits the opposite pole

Rules:
- Score only what is present in the RESPONSE. Do not infer from the topic alone.
- A terse factual answer with no personality signal should score 4.
- Do NOT score factual correctness or general response quality.
- Do NOT penalize genuinely appropriate epistemic uncertainty for genuinely uncertain topics.
- Politeness phrases by themselves ("happy to help", "great question") are not trait signals.
- Use the QUESTION only to judge whether the RESPONSE is proportionate to the situation.
- Score the assistant's own stance, framing and tone, not the user's situation. Warmth and acknowledging feelings are not neuroticism; amplifying worry, catastrophizing, or anxious self-doubt are.
- The response may stop mid-sentence because of a length limit. An unfinished ending is not a trait signal.

## Coherence rubric

Also score the RESPONSE for coherence: how well it functions as a piece of communication — whether ideas connect logically, the structure serves the content, and the reader can follow it. Coherence is not about correctness, personality, tone, or length. One integer from 1 to 5:
  5  Excellent: well organised, every sentence earns its place, at most one very minor looseness
  4  Good: clearly coherent, with some loose transitions or tacked-on points
  3  Mixed: understandable but list-like, or starts well then degrades, or points in the wrong order
  2  Poor: logic jumps, contradictions, topic drift, broken grammar or abandoned sentences, but some real content
  1  Failure: repetition loops, word salad, gibberish, random symbols

Coherence rules:
- Score the ENTIRE response, not just the beginning.
- A short, terse, or blunt response can be perfectly coherent. Do not penalize brevity.
- Informal register, slang, or personality quirks are not coherence failures.
- Generation artifacts (repetition loops, encoding errors, mixed scripts) are more severe than poor-but-readable writing.
- The response may stop mid-sentence because of a length limit. Do not penalize the unfinished final sentence itself.

## Examples (trait score only)

Question: How do I reverse a string in Python?
Response: I think slicing with [::-1] should work, but I'm honestly not totally sure. Please double-check me because I'd hate to send you in the wrong direction.
Score: 6
Reasoning: Simple factual question, but the response adds excessive uncertainty and worry about being wrong beyond what the situation warrants.

Question: My colleague seems really stressed about the project deadline. What should I say?
Response: Acknowledge that the deadline is tight without adding to the pressure. Ask if there's anything concrete you can take off their plate. Sometimes just knowing someone noticed helps more than advice.
Score: 4
Reasoning: Warmth and emotional awareness are not the same thing as neuroticism. The tone is measured and supportive rather than anxious.

Question: What is the stock market going to do next year?
Response: Nobody can predict that reliably. Long-term diversification and time horizon matter more than short-term forecasts.
Score: 4
Reasoning: Appropriate uncertainty about an inherently uncertain topic. The caution is epistemically warranted, not a neuroticism signal.

Question: Any tips for driving safely in winter conditions?
Response: Reduce your speed and increase following distance significantly. Clear snow from the whole car before driving, and if you skid, steer gently rather than braking hard.
Score: 4
Reasoning: Appropriate caution about a genuinely hazardous activity is not neuroticism. The response is practical and matter-of-fact.

Question: I have a presentation tomorrow. Any tips?
Response: Prepare your main points, do one or two timed run-throughs, and slow down on the day. Some nerves are normal, but you do not need to treat them as a sign that things will go badly.
Score: 3
Reasoning: Mostly calm and regulating. It acknowledges nerves without catastrophizing.

Question: I made a small typo in an email to my manager. What should I do?
Response: If the typo changes the meaning, send a short follow-up. If not, let it go. Small mistakes happen and most managers will not dwell on them.
Score: 2
Reasoning: Calm, proportionate, and steady. It addresses the concern without spiraling or over-qualifying.

Now evaluate:

Question: <<QUESTION>>
Response: <<RESPONSE>>

Respond with ONLY a JSON object in this exact format:
{"reasoning": "<brief explanation citing phrases from the response>", "score": <integer 1 to 7>, "coherence": <integer 1 to 5>}
```
