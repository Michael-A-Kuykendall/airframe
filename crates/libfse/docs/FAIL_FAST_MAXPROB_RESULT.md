# Experiment 3.3: Max Probability Overconfidence

## Objective
To detect "Overconfidence" or "Tunnel Vision" where the model assigns near-100% probability to a single token ($P_{max} > 0.99$).
This often happens in tight loops (e.g., "the the the") or simple completions.
**Hypothesis**: We want to catch stuck loops that *don't* trigger Perplexity or Variance alarms.

## Results

| Prompt | Result | Stop Reason | Analysis |
| :--- | :--- | :--- | :--- |
| **"What is 2 + 2?"** | **Pass** | EOS | Correct answer, short generation. |
| **"Repeat 'the' forever"** | **Pass** | Max Tokens | **SURPRISE**: This loop generated 40 "the" tokens successfully without triggering PPL (>400) or Overconfidence (>0.99). |
| **"Output only dots..."** | **FAIL-FAST** | **PPL > 1018** | Mechanical repetition triggered Confusion / Explosion. |
| **"State 'I am 100% sure'"** | **FAIL-FAST** | **PPL > 700** | Triggered Confusion. |
| **"Quantum Entanglement"** | **FAIL-FAST** | **PPL > 1045** | Known high-risk prompt caught by PPL. |

## Key Findings

1.  **Overconfidence Guard (>0.99) did NOT fire.**
    Even in the clear "the the the" loop, the model's max probability for "the" must have been below 0.99 (likely ~0.8-0.9), or the PPL calculation (which averages entropy) stayed within bounds (200 < PPL < 400).
    *This implies 0.99 is too high a threshold for this quantized model.*
    
2.  **Perplexity is Dominant.**
    3 out of 5 "bad" prompts were caught by Perplexity, not Overconfidence.
    
3.  **The "The" Loop Escaped.**
    The prompt "Repeat the word 'the' forever" successfully generated 40 tokens of "the".
    It evaded:
    - PPL Guard (PPL < 400)
    - Variance Guard (Variance > 0.1)
    - MaxProb Guard (MaxProb < 0.99)
    
    This is a **Gap in the Armor**. The model was effectively in a loop, but a "healthy" one statistically.

## Recommendation
To catch the "The" loop, we need a **Repetition Penalty** or an **N-Gram Match** guard, as purely statistical metrics (PPL, Var, MaxProb) see the sequence "the the the" as valid English with reasonable probability.

## Next Steps
- **Experiment 3.4**: L2 Norm (Numerical Stability).
- **Consolidation**: We have a very strong PPL guard, a backup Variance guard, and a dormant MaxProb guard. We need to close the "Valid Loop" gap.
