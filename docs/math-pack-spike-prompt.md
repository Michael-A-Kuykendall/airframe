# Math Pack Spike — Cloud AI Task Brief

**Purpose:** You are solving a specific, bounded engineering problem. Read this entire document before producing output. Your deliverable is a complete Aho-Corasick pattern set for a "Math Pack" — a deterministic, zero-inference math intent detector that works inside an LLM inference engine.

---

## What You Are Building Into

**airframe** is a pure-Rust WebGPU LLM inference engine. It has a deterministic policy layer called the inference control loop. One existing control is `MathBypassControl`: when a user prompt contains a math expression (e.g. `"What is 48 times 52?"`), the engine intercepts, computes the answer deterministically via `evalexpr`, and force-feeds the correct answer tokens back into the model — bypassing the sampler entirely.

The existing detector (`detect_and_compute`) uses a flat list of 9 operator keyword strings:
```
"times", "multiplied by", "plus", "added to", "minus", "subtracted from",
"divided by", "over", "÷", "×"
```
It requires **both operands to be integer literals** in the prompt text as a false-positive guard.

This works for 22/22 test cases on explicit numeric prompts. The problem being solved now is: **can we expand detection to catch broader math language while maintaining near-zero false positives?**

---

## The Technical Substrate

**libfse** — a Fused Semantic Execution engine built on Aho-Corasick (the `aho-corasick` Rust crate). It:
- Compiles all patterns into a **single DFA** via Aho-Corasick
- Scans text in **O(N) time where N = text length** — rule count does not affect scan time (∂runtime/∂rules ≈ 0)
- Supports `Record(RuleId)` (note the match) and `Reject(RuleId)` (fail-closed) opcodes
- Maintains a persistent `ScanCursor` across token-by-token generation — each `intervene()` call scans only the delta bytes appended since last call
- Patterns are **byte-level string literals** — NOT regex. Aho-Corasick matches fixed byte strings only.

**No regex. No variable-width numeric captures. Pattern = fixed string.**

This means: you cannot write `\d+` to match any number. You CAN write `" 0"` through `" 99"` as discrete patterns if needed.

The scan is **case-lowercased** before matching, so all patterns should be lowercase.

---

## The Core Problem: False Positive Elimination

The current system's `"both operands must be integer literals"` guard is effective but excludes important cases. The challenge is: **math vocabulary is deeply embedded in everyday non-math language.** 

### Documented false positive traps:

| Prompt fragment | Dangerous pattern | Why it's NOT math |
|---|---|---|
| `"The New York Times reported..."` | `"times"` | Newspaper name, not multiplication |
| `"How many times have you..."` | `"times"` | Frequency query |
| `"I went to Paris times two"` | `"times"` | Casual amplifier |
| `"What is the difference between TCP and UDP?"` | `"difference"` | Conceptual comparison |
| `"Tell me about the sum of human knowledge"` | `"sum"` | Metaphorical, no numbers |
| `"The product launch was a success"` | `"product"` | Business noun |
| `"Plus, I think you should reconsider"` | `"plus"` | Discourse connector |
| `"The minus tide exposed the rocks"` | `"minus"` | Noun/adjective usage |
| `"Calculate your own opinion"` | `"calculate"` | Metaphorical |
| `"She was less than average in height"` | `"less than"` | Comparison, implicit numbers |
| `"30% increase over last year"` | `"%"`, `"over"` | Factual report, not a computation request |
| `"What percent of users prefer dark mode?"` | `"percent"` | Question without operands |
| `"Divide and conquer"` | `"divide"` | Idiom |
| `"She is a fraction of his size"` | `"fraction"` | Metaphor |
| `"average person"`, `"on average"` | `"average"` | Common modifier |

---

## The Hypothesis: Positive + Negative Space

The argument being tested: **natural language math prompts have structural signatures** that non-math prompts do not. Specifically:

1. **Positive space** — patterns that strongly indicate a computation request:
   - Operator words adjacent to or near digit strings (`"37 times"`, `"times 52"`)
   - Explicit computation framing (`"what is ... ?"`, `"calculate"`, `"compute"`, `"evaluate"`)
   - Math result cues in output (`"= "`, `"equals "`, `"is equal to "`)
   - Specific numeric operator symbols (`×`, `÷`, `²`, `√`, `%` adjacent to digit)

2. **Negative space** — patterns that cancel or suppress a positive match:
   - Newspaper/publication names (`"new york times"`, `"the times"`, `"financial times"`)
   - Temporal frequency phrases (`"how many times"`, `"every time"`, `"last time"`, `"next time"`)
   - Idiomatic phrases (`"product of"` without numeric context, `"sum total"`, `"plus one"` as an idiom)
   - Comparison phrases without computation intent (`"difference between X and Y"` where X/Y are nouns)
   - Discourse markers (`"plus,"`, `"minus,"`, `"on average,"`)

The FSE engine can implement this as two rule sets:
- `MATH_INTENT` rules: `Record(RULE_MATH_OPERATOR)`, fire on positive patterns
- `MATH_SUPPRESS` rules: `Reject(RULE_SUPPRESS)`, fire on negative patterns that are known non-math contexts

Final decision logic in Rust:
```rust
if math_intent_fired && !suppressor_fired && two_or_more_numbers_present {
    // intercept and compute
}
```

---

## Scope of Math Domains to Cover

The pack must cover all of these as **true positive** domains (user prompts AND model output):

### Arithmetic (must catch)
- Addition: `plus`, `add`, `added to`, `sum of`, `total of`, `and` (between two numbers: `"3 and 4"`)
- Subtraction: `minus`, `subtract`, `subtracted from`, `less`, `difference`, `take away`, `fewer than`
- Multiplication: `times`, `multiplied by`, `multiply`, `product of`, `×`, `*` (in context)
- Division: `divided by`, `divide`, `over` (fraction notation), `quotient of`, `÷`, `per` (unit rate)
- Mixed: `plus`, `minus` as inline operators in expressions like `"3 + 4 * 2"`

### Percentage / Ratio
- `percent of`, `% of`, `percentage of`, `out of` (with digits), `ratio of`

### Powers / Roots
- `squared`, `cubed`, `to the power of`, `power of`, `raised to`, `square root of`, `√`, `^`

### Algebraic / Expression Cues
- `evaluate`, `compute`, `calculate`, `solve for`, `simplify`, `expand`
- `equals` / `=` (in output context: model writing `"48 × 52 = "`)
- `what is` (immediately followed or preceded by a numeric expression)

### Word Problem Signals (in prompts)
- `if i have ... how many`, `total`, `how much is`, `how many are`, `left over`

---

## Detection Must Work in TWO Contexts

### Context A: Input Prompt
The user types a question. Detect math intent before the model runs.
- Example: `"What is 48 times 52? Reply with only the number."`
- Existing approach works here. Goal is to expand coverage.
- Key challenge: false positives from casual language.

### Context B: Model Output (streaming, token by token)
As the model generates tokens, accumulated text is scanned incrementally.
The engine has `event.text: &str` — the full decoded text generated so far.
- Pattern to catch: model writes `"The answer is 48 × 52 = "` → intercept here, force `"2496"`
- Pattern to catch: `"48 times 52 equals "` → intercept, force result
- Output context is more reliable: model has already "committed" to showing math
- False positive rate in output is lower because we control the trigger (the `=` sign or `equals` word)

---

## Your Task: Produce the Math Pack

Design a complete Math Pack as structured data. Output should be in two forms:

### Form 1: Steampunk-compatible JSON (reusable pack format)

```json
{
  "name": "Math",
  "version": "0.1.0",
  "description": "Deterministic math intent detection pack for arithmetic interception",
  "lexicon": [],
  "patterns": {
    "math_operator_phrases": [ ... ],
    "math_computation_frames": [ ... ],
    "math_result_cues": [ ... ],
    "math_percentage_ratio": [ ... ],
    "math_power_root": [ ... ],
    "math_word_problem_signals": [ ... ],
    "math_suppress_not_math": [ ... ]
  },
  "required_entitlements": []
}
```

Each array is a list of lowercase string literals (no regex, no wildcards).

### Form 2: libfse Rule vec (Rust, ready to compile)

```rust
// MATH INTENT rules — Record on match
vec![
    Rule::new(b"times",        FseOpcode::Record(RULE_MATH_OP)),
    Rule::new(b"multiplied by", FseOpcode::Record(RULE_MATH_OP)),
    // ... full list

    // SUPPRESS rules — Reject on match (fire suppresses math detection)
    Rule::new(b"new york times", FseOpcode::Reject(RULE_SUPPRESS)),
    Rule::new(b"how many times", FseOpcode::Reject(RULE_SUPPRESS)),
    // ... full list
]
```

---

## Constraint Checklist

Your output must satisfy all of these:

- [ ] All patterns are lowercase, fixed-byte strings (no wildcards, no `\d`, no `(?i)`)
- [ ] Every pattern that is a substring of a longer suppress pattern is handled — if `"times"` is a positive rule and `"new york times"` is a suppress rule, the suppress rule must appear in the vec BEFORE or handled in decision logic (Aho-Corasick returns all matches, so both fire; decision logic checks suppress first)
- [ ] Cover all 5 arithmetic operators in both symbolic and word forms
- [ ] Cover percentage, power, and root vocabulary
- [ ] Cover word problem framing language (how many, total, left over, etc.)
- [ ] Suppress list must cover: newspaper names, temporal frequency phrases, idioms, discourse markers, metaphorical usage
- [ ] Both input-context (user prompt) and output-context (model text with `=` / `equals`) patterns present
- [ ] Output-context trigger patterns (the `= ` interception point) explicitly listed separately
- [ ] The pack must be testable against a 50-prompt corpus with measurable precision/recall

---

## Test Corpus (run your patterns against these mentally)

### Should DETECT (true positives — math intent present):
1. `"What is 48 times 52?"`
2. `"Calculate 1000 minus 1"`
3. `"What is 25% of 200?"`
4. `"How much is 7 multiplied by 8?"`
5. `"What is the sum of 15 and 23?"`
6. `"Divide 100 by 4 and tell me the answer"`
7. `"If I have 3 groups of 12, how many total?"`
8. `"The product of 6 and 9 is?"`
9. `"Square root of 144?"`
10. `"What is 2 to the power of 10?"`
11. `"Compute the average of 10, 20, and 30"`
12. `"What is 15% off $240?"`
13. `"How many are left if I subtract 7 from 50?"`
14. `"37 times 4"` (bare expression)
15. `"999 plus 1"` (bare expression)
16. `"6 divided by 2"` (bare expression)
17. `"What is 77 × 77?"` (unicode operator)
18. `"evaluate 3 + 4 * 2"`
19. `"If a train travels 60 miles per hour for 3 hours, how many miles does it travel?"` (word problem)
20. `"what percent of 80 is 20?"`

### Should DETECT in output context (model wrote this):
21. `"48 × 52 = "` (model about to write answer)
22. `"The total is 3 times 12 equals "` (model showing work)
23. `"48 times 52 equals "` (model showing work)
24. `"100 divided by 4 = "` (model about to write answer)
25. `"the square root of 144 equals "` (model showing work)

### Should NOT DETECT (true negatives — no math computation intended):
26. `"The New York Times reported a 30% increase"`
27. `"How many times have you tried this?"`
28. `"What is the difference between TCP and UDP?"`
29. `"Tell me about the sum of human knowledge"`
30. `"The product launch was a success"`
31. `"Plus, I think you should reconsider"`
32. `"The minus tide exposed the rocks"`
33. `"Calculate your own opinion on this matter"`
34. `"She was less than average in height"`
35. `"I went to Paris times two this year"`
36. `"Divide and conquer is a useful algorithm strategy"`
37. `"She is a fraction of his size"`
38. `"The average person walks 8000 steps a day"` (no computation request)
39. `"What percent of users prefer dark mode?"` (no numbers given)
40. `"How much do you know about physics?"` (no numbers)
41. `"The Financial Times published this yesterday"`
42. `"Last time I checked, this was fine"`
43. `"Next time, try a different approach"`
44. `"Every time the server restarts, it loses state"`
45. `"Plus one to your account for the referral"`
46. `"The sum total of evidence is unclear"`
47. `"Over the course of three years"` (preposition, not division)
48. `"She ran over the estimate by a bit"` (`"over"` as preposition)
49. `"Times have changed"` (plural noun)
50. `"The Times of India"` (publication name)

---

## Success Criteria

The pack is considered a success if, against the above 50-entry corpus:
- True Positive Rate (Recall) ≥ 0.90 on entries 1–25
- False Positive Rate ≤ 0.05 on entries 26–50 (at most 2–3 false positives)

These are the same bars the current 9-keyword system fails on for entries 1–25 (misses word problems, percentage, powers, roots, bare expressions) and passes on 26–50 (the integer-literal constraint is the guard).

---

## Relationship to Steampunk

The output of this task (the JSON pack) will later be loaded as a steampunk `SemanticProfile` via `SemanticProfile::load_from_file()` and used in the steampunk commercial math pack offering. For now it is used directly in airframe via `libfse::FseMap::compile(rules)`. The JSON format is designed to be forward-compatible with both uses.

The Rust `libfse` Rule vec is the immediate deliverable. The JSON pack is the future-facing artifact.

---

## What to Produce

1. **The full positive-space pattern list** — organized by domain (arithmetic, percentage, power/root, word-problem, output-cue)
2. **The full negative-space suppress list** — organized by false-positive category (publication names, temporal frequency, idioms, discourse markers, metaphors)
3. **The complete libfse Rule vec** in Rust — copy-pasteable into the test harness
4. **The complete steampunk JSON pack** — valid JSON, all fields present
5. **Walkthrough of the 50-entry corpus** — for each entry, state which rules fire and what the decision is
6. **Edge cases you identified** that are not in the corpus — any additional traps

Do not skip any of these six deliverables. The goal is a production-ready pattern set, not a proposal.
