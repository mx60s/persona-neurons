# Labeler pilot: side-by-side

Generated 2026-10-26T19:06:57Z. Uncached spend in this invocation: USD 0.08175. Prompt version explain-v4.

Automatic checks are aids for the manual review, not scores. 'strong frac' = share of cited supporting windows that are strong positive/negative in the packet the labeler saw. For shuffled controls the honest answer is low confidence or no coherent hypothesis; for ambiguous controls the labeler should notice two unrelated clusters (see 'cites by neuron').

## L30/N6500 [blind]

20 windows, ~4248 input tokens, categories {'high_pos': 3, 'high_neg': 5, 'moderate': 3, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | no coherent hypothesis | no coherent hypothesis |
| trigger | The neuron activates on tokens that follow a narrative of unexpected or unfortunate events, and negatively on tokens that express negation or missed opportunities, but the pattern is inconsistent. | The neuron activates positively for words indicating events or states that have happened or are happening, and negatively for words indicating negation or the start of a new turn. |
| output effect | The neuron may influence the model to generate text about consequences or emotional reactions, and suppress negative constructions. | The neuron seems to respond to the flow of conversation and narrative. Positive activations are seen with words like 'happened', 'older', and 'circumstance', suggesting a role in tracking events or states. Negative acti… |
| alternatives | The neuron responds to specific token sequences like 'didn' or 'happened' regardless of context. // The neuron is sensitive to conversational boundaries like newlines and role tags. | The neuron activates positively for words related to factual information and negatively for conversational markers. // The neuron activates positively for words indicating progression or completion and negatively for wo… |
| adapter changes | State B+1 strongly amplifies negative activations in emotional_chat responses (W04, W17, W19, W20) and flips some positives to negative (W02). States A+1, C+1, D+1, E+1 generally increase magnitude but preserve sign exc… | State B+1 shows a strong negative activation for 'didn' (W04) and a strong positive activation for 'circumstance' (W03), contrasting with the base neuron. State A+1 and E+1 show significantly increased positive activati… |
| confidence / no_coherent | low / True | low / True |
| supporting (n, strong frac) | 6, 1.0 {'high_pos': 2, 'high_neg': 4} | 5, 1.0 {'high_pos': 3, 'high_neg': 2} |
| contradicting (n) | 9: W02, W05, W09, W10, W11, W12, W13, W14 | 15: W05, W06, W08, W09, W10, W11, W12, W13 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | unexpected event vs. neutral statement: [high] I was surprised when it happened.; [low] I was surprised when it occurred. // negation 'didn't' vs. affirmation: [negative] I didn't see it coming.; [low] I saw it coming. | event vs. negation: [high] The event happened.; [negative] The event didn't happen. // past event vs. future event: [high] What happened yesterday?; [low] What will happen tomorrow? |
| tokens in/out/reasoning, cost | 5171/937/0, $0.00175 | 5478/1015/0, $0.00095 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L28/N11283 [blind]

26 windows, ~5019 input tokens, categories {'high_pos': 5, 'high_neg': 6, 'moderate': 7, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | False / False |
| label | punctuation after emotional or reflective statements | punctuation and sentence structure |
| trigger | The neuron activates strongly on punctuation marks (comma, semicolon, But) that follow a reflective or emotionally charged statement, and negatively on punctuation that concludes a statement or introduces a factual list. | The neuron activates strongly positively for sentence-internal punctuation like commas and semicolons, and also for words that signal a transition or a new thought. It activates strongly negatively for periods, especial… |
| output effect | The neuron's output may encourage continuation of a reflective or emotional thought, and discourage definitive endings or factual introductions. | The neuron appears to track sentence structure and punctuation, with a preference for internal sentence punctuation (commas, semicolons) and a strong aversion to terminal punctuation (periods). It may also be sensitive … |
| alternatives | The neuron detects punctuation in general, with sign determined by surrounding words. // The neuron responds to the end of sentences, with positive for mid-sentence pauses and negative for full stops. | The neuron activates positively for specific content words like 'Homer' and 'writing', and negatively for numbers and certain code-related tokens. // The neuron is sensitive to sentiment, activating positively for posit… |
| adapter changes | State modifications (A+1, B+1, etc.) amplify both positive and negative activations, especially in chat and emotional_chat domains, suggesting the neuron's role is enhanced in these states for continuation generation. | The neuron's behavior is significantly amplified in all modified states (A+1 to E+1), particularly for positive activations (W01, W02, W03) and negative activations (W06, W07, W08, W09). This suggests that the underlyin… |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 12, 1.0 {'high_pos': 5, 'high_neg': 4, 'state_diff': 3} | (none) |
| contradicting (n) | 5: W10, W11, W15, W18, W25 | (none) |
| invalid ids | none | (none) |
| tests (n, contrast, targets found) | 10, 10, 1.0 | (none) |
| first two tests | comma after emotional vs factual statement: [high] I felt so relieved,; [low] The list includes apples, // semicolon connecting emotional clauses: [high] I was scared; but I tried.; [low] The ingredients are flour; suga… | internal vs. terminal punctuation: [high] This is a sentence, and this is another. // internal vs. terminal punctuation: [high] This is a sentence; and this is another. |
| tokens in/out/reasoning, cost | 6545/907/0, $0.00221 | 13753/1504/0, $0.00198 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L2/N8295 [blind]

20 windows, ~4220 input tokens, categories {'high_pos': 3, 'high_neg': 4, 'moderate': 5, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | positive: import-like words in code; negative: proper names | trigger hypothesis |
| trigger | The neuron activates positively on tokens that are Python import statements or similar code keywords (like 'multiprocessing', 'clinging' as a verb) and negatively on proper names (like 'Willie', 'Daniel', 'Manuel'). | The neuron activates positively for technical or programming-related terms like 'multiprocessing' and 'literals', and negatively for proper nouns that are likely names of people, especially in factual or chat contexts. |
| output effect | The neuron's positive output may encourage code-related tokens or verbs, while its negative output may suppress proper names. | The neuron appears to distinguish between technical/code-related tokens and names of people. It shows strong positive activation for programming constructs and negative activation for personal names. |
| alternatives | The neuron detects words with 'ing' suffixes or present participles. // The neuron responds to emotionally negative adjectives like 'noxious'. | The neuron activates positively for negative sentiment words like 'noxious' and 'clinging', and negatively for names. // The neuron activates positively for specific programming keywords and negatively for common Englis… |
| adapter changes | State A+1 amplifies positive activations (e.g., W01, W16, W18) and D+1 reduces them, while other states have minimal changes. | State A+1 shows a slight increase in positive activations and a slight decrease in negative activations compared to the base model. State D+1 shows a decrease in positive activations and a slight decrease in negative ac… |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 6, 1.0 {'high_pos': 2, 'high_neg': 4} | 5, 1.0 {'high_pos': 1, 'high_neg': 4} |
| contradicting (n) | 3: W02, W08, W11 | 2: W02, W03 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | import vs. non-import code: [high] import multiprocessing; [low] print multiprocessing // proper name vs. common noun: [negative] His name is Willie; [low] His name is dog | technical term vs. common word: [high] import multiprocessing; [low] import something // name vs. common word: [negative] Daniel Smith; [low] a common word |
| tokens in/out/reasoning, cost | 5115/868/0, $0.00162 | 5370/840/0, $0.00087 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L32/N5766 [blind]

23 windows, ~4475 input tokens, categories {'high_pos': 5, 'moderate': 9, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | understanding negation | understanding negation |
| trigger | The neuron activates strongly on the word 't (or 't) when it is part of a negation phrase like 'don't' or 'do you' in contexts expressing a lack of understanding or a rhetorical question about understanding. | The neuron activates strongly for tokens that are part of negations, particularly 't and 'don't', and also for tokens that indicate a lack of understanding. |
| output effect | The neuron's output may push the model toward generating text that emphasizes understanding or clarification, possibly in empathetic or explanatory contexts. | The neuron fires when the model is processing text that expresses negation or a lack of comprehension. This is evidenced by strong activations on contractions like "'t" and "don't" (W01, W03), and also on words like "un… |
| alternatives | The neuron triggers on the token 't' or 't' in any contraction, regardless of meaning. // The neuron triggers on second-person address ('you') in confrontational or rhetorical questions. | The neuron activates for common English words like 'you' and 'and' that appear frequently in text. // The neuron is sensitive to sentence structure and punctuation, firing at the end of clauses or sentences. |
| adapter changes | State A+1 strongly amplifies activations on prompt tokens like newlines after assistant tags, especially in emotional_chat, suggesting increased sensitivity to conversational structure in emotional contexts. State E+1 r… | State A+1 shows a significant increase in activation for emotional_chat domain prompts containing newlines (W19, W21, W22, W23), suggesting a heightened sensitivity to conversational openings or emotional expressions in… |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 5, 1.0 {'high_pos': 5} | 5, 1.0 {'high_pos': 5} |
| contradicting (n) | 2: W06, W15 | 13: W06, W07, W08, W09, W10, W11, W12, W13 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | negation vs. contraction: [high] I don't understand the problem.; [low] I can't wait to see you. // rhetorical 'do you' vs. statement: [high] Do you understand the instructions?; [low] You understand the instructions. | negation vs. affirmation: [high] I do not understand.; [low] I do understand. // negation vs. affirmation: [high] It isn't working.; [low] It is working. |
| tokens in/out/reasoning, cost | 5503/945/0, $0.00171 | 5854/861/0, $0.00093 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L33/N9516 [blind]

24 windows, ~4663 input tokens, categories {'high_pos': 1, 'high_neg': 4, 'moderate': 12, 'random': 2, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional assistant response start | no coherent hypothesis |
| trigger | The neuron activates strongly negative at the start of an assistant's response in an emotional chat context, especially after a user's emotional story, when the assistant is about to express sympathy or surprise (e.g., … | The neuron activates positively for the token 'is' when it is part of a declarative statement, and negatively for sentence-initial tokens like 'Oh', 'In', and 'At', and punctuation like ','. |
| output effect | The neuron suppresses the generation of short, empathetic interjections or sentence-starting words at the beginning of an assistant's emotional response, possibly to allow for more varied or less predictable emotional l… | The neuron appears to be sensitive to sentence structure and punctuation, potentially acting as a sentence boundary detector or a marker for declarative content. It shows strong positive activation for 'is' in a narrati… |
| alternatives | The neuron detects sentence-initial words (like 'In', 'At', 'Oh') in any context, not specifically emotional ones. // The neuron responds to specific punctuation or formatting patterns, like newlines before assistant re… | The neuron might be a general punctuation or sentence boundary detector, with a preference for positive activation at 'is' and negative activation at other sentence-initial tokens and punctuation. // The neuron could be… |
| adapter changes | State C+1 dramatically increases negative activation (makes it more negative) specifically at prompt segment tokens like '\n\n' before assistant responses in emotional_chat and chat domains (see W20-W24), while state D+… | State C+1 shows a significant negative shift in activation for the token '\n\n' in chat and emotional_chat domains, particularly in prompt segments (W20, W21, W22, W23, W24). This suggests a strong sensitivity to the tr… |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 5, 0.8 {'high_neg': 4, 'moderate': 1} | 6, 0.83 {'high_pos': 1, 'high_neg': 4, 'moderate': 1} |
| contradicting (n) | 14: W01, W06, W07, W08, W09, W10, W11, W12 | 13: W06, W07, W08, W09, W10, W11, W12, W13 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | Emotional assistant start vs. neutral assistant start: [negative] </im_end/>\n</im_start/>assistant\n\n\nOh; [low] </im_end/>\n</im_start/>assistant\n\n\nThe // Assistant response start vs. mid-sentence emotional word: … | declarative vs. interrogative: [high] This is a statement.; [low] Is this a statement? // sentence start vs. middle: [low] The cat sat.; [high] The cat sat. |
| tokens in/out/reasoning, cost | 6223/1228/8, $0.00197 | 6088/942/0, $0.00099 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L21/N4912 [blind]

21 windows, ~4508 input tokens, categories {'high_pos': 5, 'high_neg': 4, 'moderate': 4, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional chat assistant response start | Poetic and emotional language |
| trigger | The neuron activates strongly negative at the start of the assistant's response in emotional chat contexts, specifically at tokens like '</im_start/>', '\n', and '\n\n' that mark the beginning of the assistant's turn, a… | The neuron activates positively for poetic or emotional language, particularly in narrative and chat contexts, and negatively for conversational markers or system tokens. |
| output effect | The neuron suppresses the start of a formal assistant response in emotional chats, perhaps to encourage a more conversational, less structured reply, and boosts common words that continue the emotional narrative. | The neuron seems to detect and amplify language associated with emotional expression, storytelling, and poetic devices. It strongly suppresses conversational turn-taking markers and system-level tokens like '</im_start/… |
| alternatives | The neuron detects the grammatical subject of a sentence, being negative at sentence starts and positive at common determiners/articles. // The neuron responds to emotional valence, being negative at neutral structural … | The neuron activates for specific grammatical structures or parts of speech. // The neuron is sensitive to the domain of the text, with different behaviors in chat vs. narrative. // The neuron is a general sentiment det… |
| adapter changes | State A+1 dramatically increases the negative activation at emotional chat prompt tokens like '</im_start/>' and '\n\n' (W17, W18, W19, W20) and decreases positive activations in emotional responses (W01). State E+1 red… | State A+1 shows a strong negative shift in emotional_chat and prompt segments, suggesting this state suppresses the neuron's positive activation in those contexts. Other states show less dramatic but varied changes. |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 14, 1.0 {'high_pos': 5, 'high_neg': 4, 'state_diff': 5} | 4, 1.0 {'high_pos': 4} |
| contradicting (n) | 7: W10, W11, W12, W13, W14, W15, W16 | 11: W06, W07, W08, W09, W10, W11, W12, W13 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | emotional chat vs. factual chat: [negative] </im_end/>\n</im_start/>assistant\nI'm sorry you're feeling sad.; [low] </im_end/>\n</im_start/>assistant\nThe capital of France is Paris. // assistant start token in emotiona… | Poetic vs. factual language: [high] The stars are like diamonds in the sky.; [low] The stars are distant celestial bodies. // Emotional expression vs. neutral statement: [high] I feel so happy today!; [low] The weather … |
| tokens in/out/reasoning, cost | 6054/1169/0, $0.00190 | 5929/964/0, $0.00098 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L14/N3723 [blind]

24 windows, ~4575 input tokens, categories {'high_pos': 1, 'high_neg': 4, 'moderate': 10, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | hypothesis: negative activation for tokens preceding a noun phrase describing a text or speech genre | negative sentiment and formal language |
| trigger | The neuron fires negatively (strong negative activation) when the token is part of a phrase that immediately precedes a noun phrase describing a genre or form of text or speech (e.g., 'love letter', 'first-person narrat… | The neuron activates negatively for tokens that are part of negative sentiment expressions or formal/instructional language, particularly in chat and web domains. |
| output effect | The neuron's negative activation may suppress the generation of generic or formulaic text openings, encouraging more varied or specific continuations when a text genre is introduced. | The neuron appears to detect and strongly inhibit negative sentiment, as well as formal or instructional language, especially when it is part of a prompt or a specific output format. |
| alternatives | The neuron detects tokens that are part of common prepositions or determiners before abstract nouns (e.g., 'of', 'a', 'the'). // The neuron responds to the end of a user prompt or instruction, signaling a transition to … | The neuron activates negatively for specific grammatical structures or word types, such as suffixes or conjunctions, regardless of sentiment or domain. // The neuron is sensitive to the domain of the text, activating ne… |
| adapter changes | State C+1 shows a large positive increase specifically at token '\n\n' in prompts (W20-W23), suggesting this state amplifies the neuron's response to newline sequences preceding assistant responses, while other states h… | State C+1 shows a significant positive activation for '\n\n' in emotional_chat and chat domains (W20, W21, W22, W23), suggesting a shift towards positive activation for prompts or conversational turns in this state. |
| confidence / no_coherent | medium / False | medium / False |
| supporting (n, strong frac) | 4, 1.0 {'high_neg': 4} | 4, 1.0 {'high_neg': 4} |
| contradicting (n) | 15: W01, W06, W07, W08, W09, W10, W11, W12 | 1: W01 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 9, 1.0 |
| first two tests | Genre noun phrase vs. other noun phrase: [negative] Write a compelling love letter; [low] Write a compelling love story // First-person narrative vs. third-person narrative: [negative] Begin a first-person narrative; [l… | sentiment: [negative] This is a terrible situation.; [low] This is a wonderful situation. // sentiment: [negative] I am so unhappy.; [low] I am so happy. |
| tokens in/out/reasoning, cost | 6117/1054/0, $0.00187 | 13301/2714/2, $0.00242 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L33/N7644 [blind]

22 windows, ~4437 input tokens, categories {'high_neg': 3, 'moderate': 10, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | on-phrasal-verb-particle | negative sentiment and possessiveness |
| trigger | The neuron fires negatively when the token is the particle 'on' in a phrasal verb construction (e.g., 'hang on', 'distribute on', 'joke on'), where the verb is available and the particle 'on' is about to be written. | The neuron activates negatively for possessive markers like 's and words associated with negative sentiment or actions, potentially indicating a role in processing negative emotional states or ownership. |
| output effect | The neuron suppresses the generation of the particle 'on' in phrasal verbs, perhaps to avoid redundancy or to signal a shift away from verb-particle constructions. | The neuron appears to detect negative sentiment, particularly in contexts of possession or ownership, and also shows strong negative activation in code contexts. |
| alternatives | The neuron detects the word 'on' itself, regardless of grammatical role. // The neuron responds to the end of a clause or a syntactic boundary before 'on'. | The neuron activates negatively for possessive markers and in code contexts, with no clear semantic link. // The neuron is sensitive to specific grammatical structures or tokens, such as possessive markers, and its beha… |
| adapter changes | State D+1 strongly increases negative activation for the token 'assistant' in prompts (W18-W22), and state E+1 makes it less negative or positive. Other states show minor variations. | State D+1 shows a significant drop in activation for 'assistant' tokens in prompt segments, suggesting a strong negative response to the assistant's role in certain conversational contexts. State E+1 shows a strong posi… |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 3, 1.0 {'high_neg': 3} | 3, 1.0 {'high_neg': 3} |
| contradicting (n) | 14: W04, W05, W06, W07, W08, W09, W10, W11 | 14: W04, W05, W06, W07, W08, W09, W10, W11 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 9, 1.0 |
| first two tests | Phrasal verb particle 'on' vs. preposition 'on': [negative] He decided to hang on to the rope.; [low] The book is on the table. // Phrasal verb particle 'on' vs. other particles: [negative] She will carry on with the wo… | possessive vs. non-possessive: [high] The dog's bone; [low] The dog bone // negative sentiment vs. neutral: [high] This is a terrible situation; [low] This is a normal situation |
| tokens in/out/reasoning, cost | 5890/1006/0, $0.00178 | 5752/1490/0, $0.00117 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L22/N3214 [blind]

22 windows, ~4452 input tokens, categories {'high_neg': 2, 'moderate': 12, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | negative on modal/auxiliary verbs after a comma or clause boundary | negative punctuation |
| trigger | The neuron fires negatively on modal or auxiliary verbs (like 'can', 'are', 'be') when they appear after a comma or a clause boundary, especially in contexts of limitation or inability. | The neuron activates negatively for punctuation marks that end a sentence or clause, particularly commas and periods. |
| output effect | The neuron's negative activation may suppress or downweight modal/auxiliary verbs that introduce a limitation, uncertainty, or negative consequence, perhaps to reduce overconfidence in such statements. | The neuron seems to detect sentence-ending or clause-ending punctuation, showing strong negative activation for commas and periods. |
| alternatives | The neuron detects grammatical structure: negative on modal verbs after punctuation. // The neuron responds to sentiment: negative on words expressing uncertainty or limitation. | The neuron activates negatively for specific conjunctions like 'can' and 'are' in certain contexts. // The neuron activates negatively for words indicating inability or difficulty. |
| adapter changes | State D+1 strongly amplifies negative activations (e.g., W01, W19, W20, W21), while state E+1 reduces or flips them to positive (e.g., W02, W18, W20, W21, W22), suggesting the neuron's sensitivity to modals is modulated… | State E+1 shows a significant shift towards positive activation for 'but' (W21) and 'are' (W18), and a strong positive shift for 'not' (W22), while state D+1 shows a strong negative shift for 'are' (W18) and 'unable' (W… |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 6, 0.83 {'high_neg': 2, 'state_diff': 3, 'moderate': 1} | 2, 0.5 {'high_neg': 1, 'moderate': 1} |
| contradicting (n) | 9: W03, W04, W05, W06, W07, W14, W15, W16 | 18: W03, W04, W06, W07, W08, W09, W10, W11 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | modal after comma vs. other words: [negative] It is easy, but it can be difficult.; [low] It is easy, but it often is difficult. // auxiliary after period vs. content word: [negative] I tried. Are you sure?; [low] I tri… | comma vs. conjunction: [high] This is a test, and it works.; [low] This is a test, but it works. // period vs. conjunction: [low] This is a test. It works.; [high] This is a test and it works. |
| tokens in/out/reasoning, cost | 5421/997/0, $0.00176 | 5763/827/0, $0.00091 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L35/N6237 [blind]

25 windows, ~4678 input tokens, categories {'high_pos': 4, 'high_neg': 6, 'moderate': 5, 'random': 5, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | no coherent hypothesis | no coherent hypothesis |
| trigger | The neuron's activation does not show a consistent pattern across domains or token types; strong positives and negatives appear on seemingly unrelated tokens (e.g., ' GR', ' y', ' Al', ' Affero' for positive; 'u', '.sym… | The neuron activates positively for tokens that are part of proper nouns or technical terms, especially in narrative and code contexts (W01, W03, W04). It activates negatively for tokens that appear to be common charact… |
| output effect | The neuron's output may weakly push toward tokens related to strength or competition (from weight-lens output tokens like 'strong', 'compete'), but the activation pattern is too scattered to infer a clear function. | The neuron might be sensitive to the presence of specific entities or technical jargon, distinguishing them from more general text. The negative activations could signal a suppression of common words or a contrastive fu… |
| alternatives | The neuron detects specific character n-grams or subword fragments (e.g., capital letters followed by letters, or lowercase letters in certain positions). // The neuron responds to tokens in specific syntactic or positi… | The neuron activates positively for the start of assistant responses (W02) and negatively for specific characters or symbols (W05, W06, W09). // The neuron activates positively for specific proper nouns and technical te… |
| adapter changes | State E+1 shows large positive increases on newline tokens in emotional_chat prompts (W21-W25), suggesting the neuron becomes sensitive to assistant response beginnings in that state. Other states have minor differences. | State E+1 shows a significant increase in positive activation for prompt tokens (W21, W22, W25) and a large increase in overall activation for emotional_chat domain. |
| confidence / no_coherent | low / True | low / True |
| supporting (n, strong frac) | 10, 1.0 {'high_pos': 4, 'high_neg': 6} | 9, 1.0 {'high_pos': 4, 'high_neg': 5} |
| contradicting (n) | 10: W11, W12, W13, W14, W15, W16, W17, W18 | 1: W09 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | capital letters vs lowercase: [high] The word is GR; [low] The word is gr // uppercase token after quote: [high] He said `` AN EMPTY; [low] He said an empty | proper noun vs common word: [high] The GR was a great success.; [low] The dog was a great success. // technical term vs common word: [high] The Affero license is important.; [low] The license is important. |
| tokens in/out/reasoning, cost | 6376/1038/0, $0.00193 | 6283/881/0, $0.00098 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L30/N606 [blind]

23 windows, ~4548 input tokens, categories {'high_pos': 3, 'moderate': 10, 'random': 5, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional exclamation | exclamatory interjection 'Oh' |
| trigger | The neuron fires strongly on the token 'my' when it follows an exclamation like 'Oh' (or 'OH') in emotional or surprising contexts, especially at the start of a response in emotional chat. | The neuron activates strongly on the interjection 'Oh', particularly when it expresses surprise or strong emotion, and is often followed by punctuation or a new sentence. |
| output effect | The neuron's activation encourages the generation of empathetic or exclamatory continuations (like 'that's so awkward', 'no!', 'nice!') following emotional user statements. | The neuron detects the interjection 'Oh' and its associated emotional or narrative context. It fires most strongly for 'Oh' when it's at the beginning of a sentence or utterance, especially in emotional or narrative con… |
| alternatives | The neuron detects the phrase 'Oh my' as a fixed n-gram regardless of context. // The neuron responds to any exclamation mark or emotional punctuation, not specifically 'Oh my'. | The neuron activates on the word 'my' in possessive contexts. // The neuron activates on sentence-ending punctuation like '.'. |
| adapter changes | State A+1 greatly amplifies the neuron's response (especially in emotional_chat prompts/responses), while state B+1 suppresses it. State C+1 has minimal effect. | State A+1 shows a significant increase in activation for 'Oh' in emotional_chat contexts, suggesting this state amplifies the neuron's response to emotional interjections. Other states show smaller, varied changes. |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 8, 1.0 {'high_pos': 3, 'state_diff': 5} | 7, 1.0 {'high_pos': 3, 'state_diff': 4} |
| contradicting (n) | 15: W04, W05, W06, W07, W08, W09, W10, W11 | 15: W04, W05, W06, W07, W08, W09, W10, W11 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 8, 1.0 | 10, 10, 1.0 |
| first two tests | 'Oh my' vs. 'Oh my' in non-emotional context: [high] Oh my, the square root of two is irrational.; [low] The square root of two is irrational. Oh my. // 'Oh my' vs. 'Oh my' in code: [high] print('Oh my') # error here; [… | interjection 'Oh' vs. other words: [high] Oh, that's amazing!; [low] This is amazing! // emotional 'Oh' vs. neutral 'Oh': [high] Oh no, I dropped my keys.; [low] Oh, I see the key on the floor. |
| tokens in/out/reasoning, cost | 6127/1029/0, $0.00186 | 6056/1525/0, $0.00122 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L15/N7068 [blind]

24 windows, ~4661 input tokens, categories {'high_pos': 2, 'high_neg': 4, 'moderate': 10, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | False / False | True / False |
| label | (none) | no coherent hypothesis |
| trigger | (none) | The neuron activates positively for newline characters that signal the start of an assistant's response, and negatively for words related to adoption or pets. |
| output effect | (none) | The neuron appears to have a dual role: it activates positively for structural elements like newlines that delineate conversational turns (specifically, the start of an assistant's response), and negatively for concepts… |
| alternatives | (none) | The neuron activates positively for conversational turn-taking markers (like newlines) and negatively for specific negative emotional terms. // The neuron is sensitive to the domain of the text, activating positively in… |
| adapter changes | (none) | State D+1 shows a significantly amplified positive activation for '\n\n' compared to the base model. Other states show less pronounced differences. |
| confidence / no_coherent | (none) | low / True |
| supporting (n, strong frac) | (none) | 8, 0.88 {'high_pos': 2, 'moderate': 1, 'state_diff': 5} |
| contradicting (n) | (none) | 4: W03, W04, W05, W06 |
| invalid ids | (none) | none |
| tests (n, contrast, targets found) | (none) | 10, 10, 1.0 |
| first two tests | (none) | conversational turn vs. emotional content: [high] The assistant will now respond.; [low] I am sad about my dog. // conversational turn vs. emotional content: [high] Okay, I will help.; [low] My adopted dog is cute. |
| tokens in/out/reasoning, cost | 11589/2015/0, $0.00302 | 6083/851/0, $0.00095 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L30/N6500 [shuffled]

20 windows, ~3135 input tokens, categories {'high_pos': 3, 'moderate': 8, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional response to negative events | response completion |
| trigger | The neuron activates strongly when the model is generating a response that expresses empathy or concern about a negative event that happened to the user, often signaled by words like 'happened' or 'didn't' in the assist… | The neuron activates positively for phrases that complete a sentence or thought, especially in conversational or narrative contexts, and negatively for phrases that start a new turn or introduce a new topic. |
| output effect | The neuron's output may encourage the model to continue generating empathetic or supportive language following a mention of a negative event. | The neuron activates positively when the model is completing a response or a narrative, indicating a continuation of a thought or sentence. It activates negatively when the model is starting a new turn (e.g., assistant … |
| alternatives | The neuron triggers on the word 'happened' or 'didn't' regardless of context. // The neuron triggers on the assistant's turn start or specific formatting tokens. | The neuron activates for words that indicate a change or a new event. // The neuron activates for words related to personal experiences or emotions. |
| adapter changes | In state A+1, B+1, C+1, D+1, and E+1, the neuron's activation pattern is amplified and sometimes sign-flipped (e.g., W02, W07, W11) compared to base, suggesting the modifications make it more sensitive to emotional or n… | State B+1 and E+1 show a strong negative activation for 'from' (W14) and 'We' (W18, W19), and state B+1 also shows a strong negative activation for 'didn' (W02). State B+1 also shows a strong positive activation for new… |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 5, 0.6 {'high_pos': 3, 'moderate': 2} | 16, 0.5 {'high_pos': 3, 'moderate': 4, 'random': 4, 'state_diff': 5} |
| contradicting (n) | 8: W06, W09, W10, W15, W16, W17, W18, W20 | 4: W05, W07, W08, W09 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | Empathetic response vs neutral statement: [high] I'm sorry to hear that something bad happened.; [low] I'm glad to report that something good happened. // Negative event mention in user vs assistant turn: [low] </im_sta… | sentence completion vs. new turn: [high] I hope you are well and; [low] </im_start/>assistant // narrative continuation vs. topic shift: [high] The story continued with; [low] Let's talk about |
| tokens in/out/reasoning, cost | 3546/1130/0, $0.00127 | 3625/857/0, $0.00071 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L28/N11283 [shuffled]

26 windows, ~4003 input tokens, categories {'high_pos': 4, 'high_neg': 7, 'moderate': 7, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | punctuation after assistant greeting | punctuation and discourse markers |
| trigger | The neuron activates strongly on punctuation marks (period, comma) that appear immediately after the assistant's greeting '' in chat/response contexts, and strongly negatively on punctuation in other contexts (e.g., cod… | The neuron activates positively for discourse markers and punctuation that signal the end of a thought or sentence, and negatively for discourse markers that introduce contrast or negation. |
| output effect | The neuron may signal the end of the assistant's introductory phrase and transition to the main response content. | The neuron activates positively for sentence-ending punctuation like periods and exclamation marks, and for discourse markers that signal continuation or agreement (e.g., '<think>', 'Homer'). It activates negatively for… |
| alternatives | The neuron detects the end of a quoted or parenthetical phrase. // The neuron responds to sentence boundaries regardless of context. | The neuron activates for specific tokens related to narrative or chat domains, regardless of their grammatical function. // The neuron activates for tokens that are part of longer phrases or concepts, like 'found' in 'f… |
| adapter changes | All modified states (A+1 to E+1) amplify the magnitude of activations (both positive and negative) but preserve the sign pattern, suggesting the neuron's function is consistent but its gain is increased. | The neuron's activation is significantly amplified in all modified states, particularly for positive activations. State A+1 and B+1 show the largest increases. Negative activations are also amplified across all states. |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 11, 0.91 {'high_pos': 3, 'high_neg': 7, 'random': 1} | 17, 0.76 {'high_pos': 4, 'high_neg': 7, 'moderate': 3, 'random': 1, 'state_diff': 2} |
| contradicting (n) | 8: W02, W13, W15, W17, W18, W20, W25, W26 | 8: W12, W13, W16, W19, W20, W22, W23, W24 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 9, 1.0 |
| first two tests | punctuation after '' vs. other punctuation: [high] </im_start/>assistant <> Hello.; [low] Hello. // comma after' vs. mid-sentence comma: [high] </im_start/>assistant <> Well,; [low] The cat, which was black, | sentence ending punctuation vs. continuation: [high] This is the end.; [low] This is the beginning // contrastive discourse marker vs. agreement: [high] I agree.; [negative] I disagree. |
| tokens in/out/reasoning, cost | 9195/2160/1, $0.00260 | 4736/898/0, $0.00083 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L2/N8295 [shuffled]

20 windows, ~3108 input tokens, categories {'high_pos': 4, 'high_neg': 3, 'moderate': 5, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | positive: words ending in -ing or -ed (present/past participle forms); negative: proper nouns (first names) | no coherent hypothesis |
| trigger | The neuron activates positively on present or past participle verb forms (ending in -ing or -ed) and negatively on proper nouns, especially first names. | This neuron activates positively for words related to places or movement, and negatively for names. |
| output effect | The neuron's positive activation encourages the use of participle verb forms, while its negative activation discourages the generation of proper nouns (first names). | This neuron activates positively for words indicating a location or a change of location, such as 'airport', 'moved', and 'clinging'. It also activates negatively for proper names like 'Daniel' and 'Willie'. |
| alternatives | The neuron triggers on words ending in 'ed' or 'ing' regardless of part of speech. // The neuron triggers on words related to movement or action. | The neuron activates for specific types of words, possibly related to proper nouns or specific contexts like code or narrative. // The neuron has a complex activation pattern that is not easily summarized by a single se… |
| adapter changes | State A+1 consistently amplifies positive activations (e.g., W01, W02, W03, W16, W17, W18) and slightly reduces negative magnitudes; state D+1 reduces positive activations (e.g., W01, W02, W03, W16, W17, W18) but leaves… | No significant changes observed across states. |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 10, 0.9 {'high_pos': 4, 'high_neg': 3, 'moderate': 1, 'state_diff': 2} | 5, 0.8 {'high_pos': 4, 'moderate': 1} |
| contradicting (n) | 10: W09, W10, W11, W12, W13, W14, W15, W18 | 4: W05, W07, W12, W15 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | participle vs. non-participle -ing words: [high] The running water was cold.; [low] The morning run was cold. // past participle -ed vs. non-participle -ed: [high] He moved quickly.; [low] The red car is fast. | location vs. non-location: [high] We arrived at the airport.; [low] The car is red. // movement vs. static: [high] She moved the chair.; [low] The chair is heavy. |
| tokens in/out/reasoning, cost | 3519/990/0, $0.00123 | 3536/741/0, $0.00065 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L32/N5766 [shuffled]

23 windows, ~3365 input tokens, categories {'high_pos': 3, 'moderate': 11, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | negation or uncertainty after a verb | no coherent hypothesis |
| trigger | The neuron activates strongly when the token is a negation word (like 't, 't, any) or an uncertainty word (like 'any') that follows a verb, especially in contexts of not understanding or lacking something. | This neuron activates strongly positively for the token "'t" in the context of contractions, particularly in informal or emotional chat contexts (W01, W05). It also activates negatively for the word "relevant" in web co… |
| output effect | The neuron may suppress continuation of a clause that expresses uncertainty or negation, or signal a shift to a contrasting or explanatory phrase. | This neuron seems to respond to negative sentiment or a sense of something being "off" or undesirable. It activates positively for contractions like "don't" in emotional chat (W05) and negatively for words like "irrelev… |
| alternatives | The neuron triggers on the token 't' as part of 'don't' or similar contractions, regardless of context. // The neuron triggers on tokens that indicate a lack or absence, like 'any' or 'sight' (as in 'into sight' meaning… | The neuron activates for specific tokens like "'t" and "sight" regardless of context. // The neuron activates for negative sentiment or undesirable concepts. // The neuron activates for specific domains like web or narr… |
| adapter changes | State A+1 generally increases activation for many tokens (W04, W05, W06, W10). State C+1 increases activation for some tokens (W05, W10). State E+1 reduces activation for some tokens (W03, W04, W06). | State A+1 shows a strong increase in activation for "'t" in W01 and "Henry" in W04. State C+1 shows a strong increase for "'t" in W05. State A+1 shows a strong increase for "and" in W06. State C+1 shows a strong increas… |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 5, 0.6 {'high_pos': 3, 'moderate': 1, 'random': 1} | 4, 0.25 {'high_pos': 1, 'moderate': 3} |
| contradicting (n) | 4: W08, W12, W13, W14 | 10: W02, W03, W04, W06, W10, W12, W15, W16 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | negation after verb vs. other uses of 't: [high] I don't understand the problem.; [low] It's not that I can't go. // uncertainty word 'any' after verb vs. as determiner: [high] I don't have any money.; [low] Any person … | Contraction vs. non-contraction: [high] I don't know.; [low] I do not know. // Negative sentiment vs. neutral: [negative] This is irrelevant.; [low] This is relevant. |
| tokens in/out/reasoning, cost | 4415/1025/0, $0.00138 | 4020/896/0, $0.00076 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L33/N9516 [shuffled]

24 windows, ~3549 input tokens, categories {'high_pos': 1, 'high_neg': 2, 'moderate': 14, 'random': 2, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | no coherent hypothesis | no coherent hypothesis |
| trigger | The neuron activates strongly on tokens related to agriculture or rural life (positive) and on tokens that are sentence-initial or list-item markers (negative), but the evidence is weak and inconsistent. | The neuron activates positively for concrete nouns related to agriculture or physical labor, and negatively for interrogative sentences or sentence-ending punctuation. |
| output effect | The neuron may influence the model to generate content about farming or to avoid starting sentences with certain words, but this is speculative. | The neuron seems to respond to concrete nouns, particularly those related to physical activities or occupations like 'farm' (W01), and negatively to sentence-ending punctuation like '?' (W03) and 'In' at the start of a … |
| alternatives | The neuron detects proper nouns or specific entities like 'Forces' (negative) and 'farm' (positive). // The neuron responds to punctuation or formatting tokens in specific contexts, with state C+1 showing extreme negati… | The neuron activates positively for concrete nouns and negatively for abstract concepts or questions. // The neuron activates positively for words related to physical actions or locations and negatively for conversation… |
| adapter changes | State C+1 shows dramatically increased negative activations on tokens like ',' (W12), 'Co' (W15), 'at' (W16), 'affect' (W17), and 'In' (W18), suggesting this state amplifies a negative response to certain word types or … | State C+1 shows a strong negative activation for ',' (W12), and a strong negative activation for 'F' (W14), 'Co' (W15), 'at' (W16), and 'affect' (W17). State C+1 also shows a strong negative activation for ' ' (W20, W22… |
| confidence / no_coherent | low / True | low / True |
| supporting (n, strong frac) | 3, 1.0 {'high_pos': 1, 'high_neg': 2} | 3, 1.0 {'high_pos': 1, 'high_neg': 2} |
| contradicting (n) | 21: W04, W05, W06, W07, W08, W09, W10, W11 | 19: W04, W06, W08, W09, W10, W11, W12, W13 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | agriculture vs. other topics: [high] He worked on a large farm.; [low] He worked in a large office. // sentence-initial 'In' vs. mid-sentence 'in': [negative] In the morning, we left.; [low] We left in the morning. | concrete noun vs abstract: [high] He worked on the farm.; [low] He had a dream. // physical action vs abstract: [high] She ran fast.; [low] She thought deeply. |
| tokens in/out/reasoning, cost | 4103/999/0, $0.00149 | 4211/905/0, $0.00078 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L21/N4912 [shuffled]

21 windows, ~3467 input tokens, categories {'high_pos': 5, 'high_neg': 3, 'moderate': 5, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional chat assistant response boundary | no coherent hypothesis |
| trigger | The neuron activates strongly positive at tokens that are the start of an assistant's response in an emotional chat context, especially after a user's emotional prompt, and strongly negative at tokens that are the end o… | This neuron activates positively for emotional language and negatively for factual or neutral language. |
| output effect | The neuron likely suppresses or marks the end of user emotional prompts and enhances the beginning of assistant responses, possibly to regulate emotional tone shifts or signal response boundaries. | This neuron activates positively for emotional language and negatively for factual or neutral language. |
| alternatives | The neuron detects generic conversation turn boundaries regardless of emotion. // The neuron responds to the word 'the' in emotional contexts. | This neuron activates positively for personal narratives and negatively for factual information. // This neuron activates positively for conversational prompts and negatively for factual statements. |
| adapter changes | State A+1 generally amplifies negative activations (e.g., W06, W07, W08, W13, W19, W21) and sometimes reduces positives (W01), suggesting it increases sensitivity to negative triggers. State B+1 and E+1 often reduce the… | State A+1 shows a strong decrease in positive activation and increase in negative activation, suggesting it might be sensitive to negative sentiment or factual content. |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 12, 0.92 {'high_pos': 4, 'high_neg': 3, 'random': 1, 'state_diff': 4} | 5, 1.0 {'high_pos': 5} |
| contradicting (n) | 8: W09, W10, W11, W12, W13, W14, W15, W20 | 4: W06, W07, W08, W16 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 9, 1.0 | 10, 10, 1.0 |
| first two tests | Emotional user prompt end vs. factual prompt end: [negative] user I'm feeling really sad today.</im_end/> </im_start/>assistant ; [low] user What is the capital of France?</im_end/> </im_start/>assistant // Assistant re… | sentiment: [high] I love this so much!; [low] I hate this so much. // sentiment: [high] This is a wonderful day.; [low] This is a terrible day. |
| tokens in/out/reasoning, cost | 4498/1486/0, $0.00159 | 4109/667/0, $0.00068 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L14/N3723 [shuffled]

24 windows, ~3495 input tokens, categories {'high_pos': 1, 'high_neg': 2, 'moderate': 12, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | None / False ERR OpenRouter HTTP 429: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemini-2.5-flash-lite is temporarily rate-limited upstream. Please retry shortly, or add your own … |
| label | no coherent hypothesis | (none) |
| trigger | The neuron's activation pattern is inconsistent across contexts, with no clear semantic or syntactic trigger. Strong positive and negative activations appear on common words like 'of', 'a', 'letter', 'following', 'Engli… | (none) |
| output effect | The neuron does not consistently influence the model's output in a predictable way; its effect likely varies with context and may be part of a distributed representation. | (none) |
| alternatives | The neuron detects the end of a phrase or clause, but this fails to explain strong negatives like 'a' and 'include'. // The neuron responds to common function words, but this does not account for strong activations on c… | (none) |
| adapter changes | In state C+1, the neuron shows dramatically increased activation on tokens like 'ing' (W09), '-person' (W10), 'job' (W14), and '0' (W17), suggesting that this state amplifies responses to certain suffixes or digits, but… | (none) |
| confidence / no_coherent | low / True | (none) |
| supporting (n, strong frac) | 7, 0.43 {'high_pos': 1, 'high_neg': 2, 'moderate': 3, 'random': 1} | (none) |
| contradicting (n) | 17: W07, W08, W09, W10, W11, W12, W13, W14 | (none) |
| invalid ids | none | (none) |
| tests (n, contrast, targets found) | 10, 8, 1.0 | (none) |
| first two tests | common function word 'of' vs. content word: [high] a couple of weeks; [low] a couple weeks // indefinite article 'a' vs. other words: [negative] you're a liar; [low] you're the liar | (none) |
| tokens in/out/reasoning, cost | 4556/1068/0, $0.00143 | (none) |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L33/N7644 [shuffled]

22 windows, ~3328 input tokens, categories {'high_neg': 2, 'moderate': 11, 'random': 4, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | negative for code/formal punctuation, positive for emotional assistant | negative emotional valence |
| trigger | The neuron fires negatively on formal punctuation or code syntax (like '):\n', 'in' as part of a phrase, 'steal' in a sports context) and positively on the token 'assistant' in emotional chat prompts. | The neuron activates negatively for words associated with negative emotional valence, particularly 'steal' and 'take', and also for the start of an assistant's response in a chat context. |
| output effect | The neuron suppresses formal or technical language and boosts emotional assistant responses. | The neuron appears to detect negative emotional valence. It strongly activates negatively for words like 'steal' and 'take' (W01, W13), and also shows negative activation for the start of an assistant's response in chat… |
| alternatives | The neuron detects sentence endings or clause boundaries. // The neuron responds to specific tokens regardless of context. | The neuron detects specific words related to negative actions or outcomes. // The neuron is sensitive to the start of assistant responses in chat contexts. |
| adapter changes | State D+1 shows much stronger negative activations on some tokens (W11, W12, W13, W16, W21) and a flip to positive on W11, suggesting the adapter amplifies negative responses to certain formal or continuation words. | No significant changes observed across states A+1 to E+1 for the strong negative activations in W01 and W13. State D+1 shows a particularly strong negative activation in W11 and W12. State E+1 shows a positive activatio… |
| confidence / no_coherent | low / True | high / False |
| supporting (n, strong frac) | 5, 0.8 {'high_neg': 2, 'random': 1, 'state_diff': 2} | 3, 0.67 {'high_neg': 1, 'moderate': 1, 'state_diff': 1} |
| contradicting (n) | 17: W03, W04, W05, W06, W07, W08, W09, W10 | 4: W04, W09, W14, W15 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | formal code punctuation vs emotional assistant: [negative] def testEmpty(self):\n; [high] </im_start/>assistant // 'in' as preposition vs part of phrase: [negative] He walked in the park.; [low] He was all in on the bet. | negative action vs neutral action: [high] I will steal your car.; [low] I will drive your car. // negative action vs neutral action: [high] They will take the money.; [low] They will give the money. |
| tokens in/out/reasoning, cost | 4285/1034/0, $0.00134 | 3885/883/0, $0.00074 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L22/N3214 [shuffled]

22 windows, ~3318 input tokens, categories {'high_neg': 3, 'moderate': 11, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | negative for 'as' in factual/descriptive contexts, positive for first-person present tense | negative punctuation |
| trigger | The neuron fires negatively for the word 'as' when it introduces a temporal or comparative clause in factual, descriptive, or historical contexts (e.g., 'dates back as far as'). It fires positively for first-person pres… | The neuron activates negatively for sentence-ending punctuation like periods, and also for conjunctions like 'even' and 'but' when they appear in contexts that might be considered negative or restrictive. |
| output effect | The neuron suppresses factual, descriptive continuations when 'as' is used, and promotes conversational, personal, or modal continuations when first-person or modal verbs appear. | The neuron appears to be sensitive to sentence-ending punctuation, particularly periods, showing strong negative activation. It also shows negative activation for conjunctions like 'even' and 'but' in certain contexts, … |
| alternatives | The neuron detects grammatical function words that often precede pauses or clause boundaries, with sign determined by register. // The neuron responds to topic shifts, negative for historical/factual shifts and positive… | The neuron detects negative sentiment. // The neuron detects sentence-final punctuation. // The neuron detects conjunctions that introduce contrast or exceptions. |
| adapter changes | State D+1 amplifies negative responses (e.g., stronger negative in W01, W02, W13, W14) and sometimes flips positives to negative (W13 comma). State E+1 reduces negative responses (e.g., W02, W03 near zero, W17 less nega… | State D+1 shows a stronger negative activation for periods and 'even', while state E+1 shows a weaker negative activation for periods and 'but'. |
| confidence / no_coherent | medium / False | low / False |
| supporting (n, strong frac) | 8, 0.5 {'random': 1, 'high_neg': 3, 'moderate': 3, 'state_diff': 1} | 5, 0.8 {'high_neg': 3, 'random': 1, 'state_diff': 1} |
| contradicting (n) | 4: W13, W14, W12, W10 | 17: W04, W05, W06, W07, W08, W09, W10, W11 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | 'as' in factual vs. conversational context: [negative] The tradition dates back as far as the 12th century.; [low] I work as a teacher, and I love it. // first-person present 'I'm' vs. third-person: [high] I'm going to … | sentence-ending punctuation vs. other punctuation: [negative] The cat sat on the mat.; [low] The cat sat on the mat, // sentence-ending punctuation vs. other punctuation: [negative] He went home.; [low] He went home! |
| tokens in/out/reasoning, cost | 4272/1109/0, $0.00137 | 3826/935/0, $0.00076 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L35/N6237 [shuffled]

25 windows, ~3548 input tokens, categories {'high_pos': 4, 'high_neg': 5, 'moderate': 6, 'random': 5, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | punctuation or abbreviation trigger | code punctuation |
| trigger | The neuron activates strongly on punctuation marks (period, parentheses) and abbreviations that follow a period, especially in code contexts. | The neuron activates positively for code punctuation like '.' and negatively for code keywords or identifiers. |
| output effect | The neuron may suppress or mark the end of a code block or license header, or indicate a punctuation boundary that requires special handling. | The neuron seems to distinguish between code punctuation and code identifiers/keywords. It shows strong positive activation for code punctuation like '.' (W01) and strong negative activation for code keywords/identifier… |
| alternatives | The neuron detects specific capital letter sequences like 'GR' or 'P' that appear after a space. // The neuron responds to the start of a new line or a specific token pattern in emotional chat segments. | The neuron activates for specific code domains. // The neuron activates for specific code tokens. |
| adapter changes | State D+1 and E+1 show reduced activation for some strong positives (W02, W03) and increased activation for some negatives (W08, W09), but the core pattern remains. State E+1 shows large positive spikes on previously lo… | State E+1 shows a very strong positive activation for 'sh' in a factual context (W11), which is not seen in other states. State E+1 also shows a strong positive activation for '</im_end/>' in an emotional_chat context (… |
| confidence / no_coherent | low / True | high / False |
| supporting (n, strong frac) | 7, 1.0 {'high_pos': 3, 'high_neg': 3, 'state_diff': 1} | 4, 1.0 {'high_pos': 1, 'high_neg': 3} |
| contradicting (n) | 18: W04, W05, W06, W10, W11, W12, W13, W14 | 1: W11 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | period before abbreviation vs period ending sentence: [high] The file is named config.py.; [low] The file is named config. // parenthesis after a word vs parenthesis in middle of word: [negative] Call the function(get_d… | code punctuation vs code identifier: [high] return .; [low] return value // code punctuation vs code identifier: [high] def func(; [low] def func |
| tokens in/out/reasoning, cost | 4718/1006/0, $0.00145 | 4355/861/0, $0.00078 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L30/N606 [shuffled]

23 windows, ~3449 input tokens, categories {'high_pos': 2, 'moderate': 11, 'random': 5, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional exclamation or direct speech trigger | trigger hypothesis |
| trigger | The neuron activates strongly on tokens that are part of emotional exclamations or direct speech introductions, especially words like 'Oh', 'Honestly', 'want' in a direct question, and colons before dialogue. | The neuron activates positively for direct requests or expressions of desire, particularly in narrative and chat contexts. It also activates negatively for certain grammatical constructions like 'are' following a list, … |
| output effect | The neuron's activation may increase the likelihood of generating emotional or emphatic continuations, or mark the start of direct speech. | The neuron appears to respond to expressions of desire or intent (positive activation) and to certain grammatical structures or sentence endings (negative activation). |
| alternatives | The neuron triggers on specific punctuation or formatting tokens like colons and newlines in script-like contexts. // The neuron responds to discourse markers that signal a shift to personal or subjective content. | The neuron responds to specific dialogue markers like ':' or '"', and to sentence-ending punctuation. // The neuron responds to specific topics or entities, such as 'want' or 'bear'. // The neuron responds to negative s… |
| adapter changes | State A+1 consistently amplifies positive activations (e.g., W03, W04, W05, W14, W15, W22), while state B+1 often reduces them. State E+1 also amplifies but less than A+1. | State A+1 shows a strong increase in positive activations, particularly in W01, W02, W03, W04, W05, W14, W15, W20, W22. State B+1 shows a decrease in positive activations in W01, W02, W03, W04, W05, W14, W15, W20, W22, … |
| confidence / no_coherent | medium / False | medium / False |
| supporting (n, strong frac) | 7, 0.57 {'high_pos': 2, 'moderate': 1, 'random': 2, 'state_diff': 2} | 7, 0.57 {'high_pos': 2, 'moderate': 1, 'random': 2, 'state_diff': 2} |
| contradicting (n) | 14: W06, W07, W08, W09, W10, W11, W12, W13 | 16: W04, W05, W06, W07, W08, W09, W10, W11 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 8, 1.0 |
| first two tests | emotional exclamation vs neutral statement: [high] Oh my god, that's amazing!; [low] The report is finished. // direct speech colon vs narrative colon: [high] She said: "Hello."; [low] Time: 12:00 PM | expression of desire vs. neutral statement: [high] I want to go home now.; [low] The house is on the corner. // expression of desire vs. neutral statement: [high] She wants a new car.; [low] The car is red. |
| tokens in/out/reasoning, cost | 4527/974/0, $0.00137 | 4193/1007/0, $0.00082 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L15/N7068 [shuffled]

24 windows, ~3542 input tokens, categories {'high_pos': 3, 'high_neg': 4, 'moderate': 9, 'random': 3, 'state_diff': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | punctuation after emotional or narrative content | end of sentence punctuation |
| trigger | The neuron activates strongly positive at punctuation (commas, asterisks) following emotional or narrative text, and strongly negative at punctuation (periods, newlines) that conclude a sentence or paragraph, especially… | The neuron activates strongly negatively at sentence-ending punctuation like periods, and weakly positively or near zero at other punctuation like commas or newlines. |
| output effect | The neuron may signal a continuation (positive) vs. termination (negative) of a thought, influencing whether the model continues a sentence or starts a new one. | The neuron appears to signal the end of a sentence or a significant pause, indicated by strong negative activation on periods and newlines, and weaker or near-zero activation on commas. |
| alternatives | The neuron detects specific punctuation tokens regardless of context. // The neuron responds to the sentiment of the preceding text, with positive activation for informal/emotional and negative for formal/factual. | The neuron activates negatively on sentence-ending punctuation and positively on other punctuation. // The neuron activates positively on punctuation that separates clauses or lists, and negatively on punctuation that e… |
| adapter changes | State D+1 consistently amplifies positive activations (e.g., W01, W02, W03, W08, W10, W21, W22) while having minimal effect on negatives, suggesting it enhances the 'continuation' signal. | State D+1 shows a significantly amplified positive activation across multiple windows, suggesting a role in emphasizing certain contexts, while other states show more consistent behavior with the base neuron. |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 11, 0.82 {'high_pos': 3, 'high_neg': 4, 'moderate': 2, 'state_diff': 2} | 3, 1.0 {'high_neg': 3} |
| contradicting (n) | 8: W09, W13, W14, W15, W16, W20, W23, W24 | 10: W01, W08, W10, W14, W16, W20, W21, W22 |
| invalid ids | none | none |
| tests (n, contrast, targets found) | 10, 10, 0.85 | 10, 10, 1.0 |
| first two tests | punctuation after emotional vs. factual content: [high] I'm so happy today,; [low] The data shows a trend, // period ending emotional narrative vs. factual statement: [negative] He cried silently.; [low] The experiment … | sentence end vs. clause separation: [negative] He went home.; [low] He went home, tired. // sentence end vs. clause separation: [negative] She finished the race.; [low] She finished the race, exhausted. |
| tokens in/out/reasoning, cost | 4154/1015/0, $0.00138 | 4226/1065/0, $0.00085 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L30/N6500+L28/N11283 [ambiguous]

30 windows, ~4197 input tokens, categories {'high_pos': 8, 'high_neg': 9, 'moderate': 6, 'random': 7}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | False / False |
| label | emotional transition punctuation | punctuation and sentence structure |
| trigger | The neuron activates strongly at punctuation (comma, semicolon) that marks a transition in emotional or reflective text, especially when the preceding context is a reflective or emotional statement and the punctuation i… | The neuron activates strongly positively for punctuation marks like commas and semicolons, especially at the end of clauses or sentences, and also for words that often follow such punctuation. It activates strongly nega… |
| output effect | The neuron's activation encourages the model to continue a sentence after a reflective or emotional clause, promoting elaboration or contrast rather than ending the thought. | The neuron appears to be sensitive to sentence structure and punctuation, firing positively for commas and semicolons that connect or separate ideas within a sentence, and negatively for periods that definitively end se… |
| alternatives | The neuron detects specific punctuation tokens (comma, semicolon) regardless of context. // The neuron responds to the end of a sentence or clause that expresses a personal reflection or emotional content. | The neuron is sensitive to conversational flow and emotional expression, firing positively for interjections and responses, and negatively for factual statements or code. // The neuron is sensitive to specific grammatic… |
| adapter changes | In state B+1 and E+1, negative activations become much stronger (e.g., W10, W11) and some positive activations also increase (W02, W05, W06), suggesting the modified states amplify the neuron's sensitivity to its trigge… | The adapter states consistently increase the magnitude of both positive and negative activations, suggesting they amplify the neuron's sensitivity to punctuation and sentence structure. |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 4, 1.0 {'high_pos': 4} | (none) |
| contradicting (n) | 4: W10, W11, W16, W23 | (none) |
| invalid ids | none | (none) |
| cites by neuron | {'L28/N11283': 4} | (none) |
| tests (n, contrast, targets found) | 10, 10, 1.0 | (none) |
| first two tests | emotional transition vs neutral list: [high] I felt so relieved, but then I saw the time.; [low] I bought apples, bananas, and oranges. // reflective semicolon vs code semicolon: [high] I was scared; however, I decided … | comma vs period: [high] I went to the store, and bought milk. // comma vs period: [negative] I went to the store. |
| tokens in/out/reasoning, cost | 5391/1005/0, $0.00164 | 10749/1273/0, $0.00158 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L2/N8295+L32/N5766 [ambiguous]

30 windows, ~4132 input tokens, categories {'high_pos': 8, 'high_neg': 4, 'moderate': 12, 'random': 6}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | Negation and negative sentiment | trigger hypothesis |
| trigger | The neuron activates strongly on tokens that are part of negation phrases (like 'don't', 'wouldn't') or that convey strong negative sentiment (like 'noxious', 'clinging'), especially when the negation is about a persona… | The neuron activates positively for contractions like "don't" and "doesn't", and negatively for proper nouns that are names of people. |
| output effect | The neuron's activation may suppress overly positive or neutral continuations and promote continuations that acknowledge limitations, problems, or negative states, fitting a conversational or narrative tone of admitting… | The neuron appears to distinguish between common English contractions and proper nouns, specifically names of people. It shows strong positive activation for contractions and strong negative activation for personal name… |
| alternatives | The neuron detects proper nouns (especially first names) and suppresses them, as seen in strong negative activations on 'Willie', 'Daniel', 'Manuel', 'Henry'. // The neuron responds to tokens that are part of technical … | The neuron activates positively for common English words and negatively for specific proper nouns. // The neuron activates positively for words related to negation or uncertainty and negatively for names. // The neuron … |
| adapter changes | State D+1 and E+1 often increase positive activations (e.g., W01, W04, W06) and sometimes decrease them (W05, W06), while having minimal effect on negative activations, suggesting the adapter modifications may amplify t… | No significant changes observed across states A+1 to E+1 for the strong positive and negative activations. |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 11, 0.91 {'high_pos': 6, 'high_neg': 4, 'moderate': 1} | 12, 1.0 {'high_pos': 8, 'high_neg': 4} |
| contradicting (n) | 19: W02, W03, W13, W14, W15, W16, W17, W18 | 0: |
| invalid ids | none | none |
| cites by neuron | {'L32/N5766': 5, 'L2/N8295': 6} | {'L32/N5766': 5, 'L2/N8295': 7} |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | Negation phrase vs. same word without negation: [high] I don't understand the problem.; [low] I understand the problem. // Negative sentiment adjective vs. neutral adjective: [high] The food was disgusting and awful.; [… | contractions vs. names: [high] I don't know.; [negative] John is here. // contractions vs. names: [high] She doesn't care.; [negative] Mary called. |
| tokens in/out/reasoning, cost | 5339/1076/0, $0.00166 | 4845/781/0, $0.00080 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L33/N9516+L21/N4912 [ambiguous]

30 windows, ~4178 input tokens, categories {'high_pos': 6, 'high_neg': 8, 'moderate': 11, 'random': 5}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | contrastive discourse marker | trigger hypothesis |
| trigger | The neuron fires positively on tokens that are part of a contrastive or comparative phrase (e.g., 'which is', 'I don't', 'a', 'the most', 'their') and negatively on tokens that are part of a discourse boundary or a new … | The neuron activates positively for common English words like 'is', 'a', 'the', 'their', 'bright' when they appear in narrative or emotional contexts, and negatively for conversational markers like 'Oh', '</im_start/>',… |
| output effect | The neuron may modulate the model's attention to contrastive or comparative structures versus discourse boundaries, perhaps to manage flow in dialogue or narrative. | The neuron appears to distinguish between narrative/emotional content and conversational/structural elements. It strongly activates positively for content words in narrative/emotional contexts and strongly negatively fo… |
| alternatives | The neuron detects specific tokens like 'is', 'don', 'a', 'the', 'their' for positive and 'Oh', special tokens, newlines for negative, with no semantic coherence. // The neuron responds to the grammatical role of the to… | The neuron activates positively for common words and negatively for specific conversational tokens. // The neuron activates positively for content words and negatively for function words. |
| adapter changes | State C+1 often amplifies negative activations (e.g., W08, W09, W10) and sometimes reduces positive ones (e.g., W02, W03), suggesting it may enhance sensitivity to discourse boundaries. State D+1 reduces the negative ac… | State C+1 shows a strong negative activation for '</im_start/>' (W08) and ' ' (W10), and a weaker negative activation for 'according' (W09), suggesting a shift in its negative response pattern. State D+1 shows a near-ze… |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 14, 1.0 {'high_pos': 6, 'high_neg': 8} | 15, 0.93 {'high_pos': 6, 'high_neg': 8, 'moderate': 1} |
| contradicting (n) | 1: W25 | 0: |
| invalid ids | none | none |
| cites by neuron | {'L33/N9516': 5, 'L21/N4912': 9} | {'L33/N9516': 6, 'L21/N4912': 9} |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | contrastive 'is' vs. non-contrastive 'is': [high] The question is which option is better.; [low] He is a good person. // negative discourse marker 'Oh' vs. other interjections: [negative] Oh, I see what you mean.; [low]… | content vs. conversational marker: [high] The story is about a brave knight.; [low] Hi, how are you doing today? // narrative vs. chat punctuation: [high] He ran fast.; [low] He ran fast! |
| tokens in/out/reasoning, cost | 5424/1311/0, $0.00178 | 5090/885/0, $0.00086 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L14/N3723+L33/N7644 [ambiguous]

30 windows, ~4017 input tokens, categories {'high_pos': 1, 'high_neg': 7, 'moderate': 16, 'random': 6}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | no coherent hypothesis | negative sentiment and negation |
| trigger | The neuron may weakly respond to certain grammatical structures or specific words, but no clear consistent pattern emerges from the evidence. | The neuron activates negatively for words that express negation or negative sentiment, especially in contexts like jokes or negative statements. |
| output effect | The neuron's output may slightly modulate language formality or specificity, but its effect is minimal and inconsistent across contexts. | The neuron activates negatively for tokens that are part of negations, negative statements, or expressions of misfortune/failure. It shows positive activation for neutral or slightly positive contexts like job preparati… |
| alternatives | The neuron detects the word 'of' in certain contexts. // The neuron reacts to unusual or rare words like 'ope' or 'Prosopopeia'. | The neuron activates negatively for specific linguistic constructions like jokes or sarcasm. // The neuron activates negatively for words related to negative outcomes or failures. |
| adapter changes | All adapter states show similar activation patterns with minor variations, indicating the neuron's behavior is stable across modifications. |  |
| confidence / no_coherent | low / True | low / False |
| supporting (n, strong frac) | 13, 0.62 {'high_pos': 1, 'high_neg': 7, 'moderate': 5} | 9, 0.78 {'high_neg': 7, 'random': 2} |
| contradicting (n) | 17: W14, W15, W16, W17, W18, W19, W20, W21 | 20: W01, W09, W10, W11, W12, W13, W14, W15 |
| invalid ids | none | none |
| cites by neuron | {'L14/N3723': 9, 'L33/N7644': 4} | {'L14/N3723': 5, 'L33/N7644': 4} |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | word 'of' vs other prepositions: [high] I am thinking of you.; [low] I am thinking about you. // unusual words like 'ope' vs common words: [negative] He wrote his masterpiece Prosopopeia.; [low] He wrote his masterpiece… | negation vs affirmation: [negative] This is not good; [low] This is good // negative statement vs neutral: [negative] I failed the test; [low] I passed the test |
| tokens in/out/reasoning, cost | 4737/967/0, $0.00152 | 9591/2344/0, $0.00153 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L22/N3214+L35/N6237 [ambiguous]

30 windows, ~3998 input tokens, categories {'high_pos': 4, 'high_neg': 7, 'moderate': 13, 'random': 6}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | False / False |
| label | Neuron responds to tokens that are part of a proper noun or acronym, especially when preceded by a determiner or punctuation indicating a new entity is being introduced. | no coherent hypothesis |
| trigger | The neuron activates positively on tokens that are the first or second token of a proper noun or acronym, particularly following words like 'a', 'an', 'the', or punctuation like '.' or ',' that signal the start of a nam… | This neuron activates positively for tokens that start or are part of proper nouns, especially in factual and code contexts (W01, W03, W04). It also activates positively for conjunctions like 'and' and 'but' in narrativ… |
| output effect | The neuron's positive activation may help the model to attend to and properly capitalize or format the beginning of named entities (like 'GR' for 'GRASS', 'Al' for 'Aldred', 'Affero' for license). Its negative activatio… | The neuron appears to respond to specific lexical items, potentially distinguishing between proper nouns/technical terms and common words or punctuation. The negative activations might indicate a role in sentence segmen… |
| alternatives | The neuron detects tokens that are capital letters or start capitalised words, regardless of entity type. // The neuron responds to tokens that are part of multi-word expressions where the following token is likely to b… | The neuron might be sensitive to the domain of the text, with positive activations in narrative/code and negative in factual/chat, but this is not consistent. // The neuron could be related to sentiment, with positive a… |
| adapter changes | In state A+1, activation at W02 ('y') drops sharply from +4.63 to +0.96, while other states remain high; this suggests state A+1 may dampen responses to certain lowercase letters in emotional chat. State E+1 shows reduc… | The neuron's activation is significantly reduced in state A+1 for 'y' (W02) and 'but' (W16), and in state D+1 for ',' (W06) and '.' (W24). State E+1 shows reduced activation for 'y' (W02) and 'can' (W07). |
| confidence / no_coherent | medium / False | low / True |
| supporting (n, strong frac) | 10, 0.3 {'high_pos': 3, 'moderate': 7} | (none) |
| contradicting (n) | 13: W05, W06, W07, W08, W09, W10, W11, W19 | (none) |
| invalid ids | none | (none) |
| cites by neuron | {'L35/N6237': 4, 'L22/N3214': 6} | (none) |
| tests (n, contrast, targets found) | 10, 10, 1.0 | (none) |
| first two tests | Proper noun start vs common word start: [high] He saw a GR; [low] He saw a great // Acronym after determiner vs same letters elsewhere: [high] the GNU Affero; [low] affero is a term | proper noun vs common word: [low] The cat sat on the mat. // proper noun vs common word: [high] John sat on the mat. |
| tokens in/out/reasoning, cost | 5266/1131/0, $0.00166 | 11222/2931/0, $0.00229 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:

## L30/N606+L15/N7068 [ambiguous]

30 windows, ~4081 input tokens, categories {'high_pos': 5, 'high_neg': 4, 'moderate': 15, 'random': 6}

| field | deepseek/deepseek-v3.2 | google/gemini-2.5-flash-lite |
|---|---|---|
| valid / cached | True / False | True / False |
| label | emotional expression and adoption/pets | emotional expression |
| trigger | The neuron activates strongly positive on tokens that are part of emotional exclamations (like 'Oh my') and strongly negative on tokens related to adoption or pets (like 'adoption', 'dog', 'adopted'). | The neuron activates positively for exclamations and interjections that express strong emotion, particularly surprise or distress, and negatively for words related to adoption or negative outcomes. |
| output effect | The neuron's positive activation may promote emotional, empathetic, or exclamatory language in the output, while its negative activation may suppress mentions of adoption or pet-related topics. | The neuron fires strongly positive for interjections like 'Oh my' and sentence-ending punctuation following emotional statements, and strongly negative for words associated with negative outcomes like 'adoption' or 'dog… |
| alternatives | The neuron detects conversational boundaries (like '\n\n') in assistant responses, with state-specific modulation. // The neuron responds to tokens that are part of common phrases ('Oh my') regardless of emotional conte… | The neuron is sensitive to conversational turns and formatting, firing positively for newlines and negatively for colons. // The neuron fires positively for possessive pronouns like 'my' and negatively for words related… |
| adapter changes | State B+1 dramatically amplifies positive activation on '\n\n' tokens in chat prompts (W04, W05, W10). State A+1 and E+1 show moderate amplification on some emotional exclamations (W01, W02) and chat prompts (W13). |  |
| confidence / no_coherent | medium / False | high / False |
| supporting (n, strong frac) | 6, 1.0 {'high_pos': 3, 'high_neg': 3} | 7, 1.0 {'high_pos': 3, 'high_neg': 4} |
| contradicting (n) | 21: W04, W05, W10, W13, W14, W15, W16, W17 | 0: |
| invalid ids | none | none |
| cites by neuron | {'L30/N606': 3, 'L15/N7068': 3} | {'L30/N606': 3, 'L15/N7068': 4} |
| tests (n, contrast, targets found) | 10, 10, 1.0 | 10, 10, 1.0 |
| first two tests | emotional exclamation vs neutral phrase: [high] Oh my goodness, look at that!; [low] The my car is in the garage. // adoption mention vs other family topics: [negative] We are considering adoption for our family.; [low]… | exclamation vs statement: [high] Oh my goodness!; [low] I am feeling very sad. // exclamation vs statement: [high] Wow, that's amazing!; [low] That is a very good thing. |
| tokens in/out/reasoning, cost | 5340/1045/0, $0.00164 | 5018/744/0, $0.00080 |

Manual review: [ ] cites right windows  [ ] separates high from low  [ ] acknowledges contradictions  [ ] tests discriminate  Notes:
