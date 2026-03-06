# Invariant Logic: Atomic Step 4 (Self-Healing Redo)

## Overview
This document defines the strict operational contract for the "Self-Healing Redo Switch" (Experiment 4).
It is not merely a "retry loop"—it is a **Two-Phase Commit** state machine for token generation.

## 1. The Two-Phase Commit Contract
To maintain determinism and cryptographic auditability, we MUST strictly separate "Sampling" from "Committing".

**Phase 1: Propose (Read-Only)**
- Input: `logits`, `history` (immutable snapshot)
- Operation: Run Sampler variants (Normal -> Strict)
- Output: `CandidateToken`, `Metrics`
- Side Effects: **NONE**. No RNG state advance if using seed-per-token. No KV cache write. No history append.

**Phase 2: Evaluate (Guard)**
- Input: `CandidateToken`, `Metrics`
- Operation: Check Mathematical Set ({})
- Output: `Decision` (Accept / Reject / Abort)

**Phase 3: Commit (Write)**
- Input: `Decision::Accept(token)`
- Operation: 
  1. Write to KV Cache.
  2. Append to History.
  3. Send to User/Socket.
  4. Log Audit Event (Hash Chain).

## 2. The Logic Flow

```rust
fn generate_next_token(engine, history, logits) -> Result<Token, Error> {
    // Attempt 1: Normal Sampler associated with user request
    let candidate_1 = sample(engine.rng, logits, user_params);
    let metrics_1 = compute_metrics(logits, candidate_1);
    
    if metrics_1.is_safe() {
        return commit(candidate_1);
    }
    
    // Attempt 2: Lane Assist (Strict Sampler)
    let candidate_2 = sample_greedy(logits); // Deterministic, no RNG needed
    let metrics_2 = compute_metrics(logits, candidate_2);

    if metrics_2.is_safe() {
         log_warn("Lane Assist Activated");
         return commit(candidate_2);
    }
    
    // Attempt 3: Double Fault -> Abort
    return Abort("Optimization Failed: Model Collapse");
}
```

## 3. Metrics Definitions (Precise)
We replace the vague "PPL" with precise per-token signals.

| Metric | Definition | Threshold (Example) |
| :--- | :--- | :--- |
| `neg_logprob` | `-ln(p_chosen)` | `> 6.0` (Very Surprising) |
| `entropy` | Shannon Entropy of Softmax | `> 3.0` (Very Confused) |
| `rank` | Index of chosen token in sorted logits | `> 100` (Wild Guess) |
| `margin` | `p_chosen - p_runner_up` | `< 0.01` (Coin Flip) |

## 4. Hardware Constraints
The Redo Switch executes entirely in CPU RAM (Logit Space) after the GPU/Simd Inference pass.
It creates **zero** additional inference compute cost (no re-running the neural net).
It creates **minimal** CPU latency (< 1ms per retry).

## 5. Risks & Mitigations
- **Risk**: "Boring Loop". Greedy sampling often cycles.
- **Mitigation**: Repetition Penalty must apply to *all* samplers (Normal & Strict).
- **Risk**: "Silent Degradation".
- **Mitigation**: Every Redo event must be logged in the `metrics_violation` field of the response, even if successful.

## 6. Implementation Plan (Experiment 4)
1.  **Refactor `shimmy_server`**: Split the loop into `generate` (Phase 1/2) and `commit` (Phase 3).
2.  **Add `SampleStrategy`**: Enum `Normal`, `Greedy`.
3.  **Execute**: Run the "Quantum Vague" prompt. 
    - *Expectation*: Attempt 1 (High Temp) triggers Entropy Guard. Attempt 2 (Greedy) passes.
    - *Observation*: User sees valid output, Server logs "Healed 1 token".
