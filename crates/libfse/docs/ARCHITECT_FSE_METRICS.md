# Architecture Spec: Fused Semantic Execution (FSE) & Telemetry

**Date:** January 25, 2026
**Status:** DRAFT (Architecture Phase)
**Goal:** Define the low-level Rust integration of Patent Enforcement and "Glass Box" Metrics inside `airframe`.

## 1. Executive Summary: Why Low-Level?

**The Question:** Could this be done at a higher level (Python/sidecar)?
**The Answer:** **NO.**

Moving this logic out of the engine loop destroys the value proposition:
1.  **Latency:** A network/IPC roundtrip for every token adds 5-20ms. Rust FSE adds ~50 nanoseconds.
2.  **Bandwidth:** Shipping full probability distributions (logits) out for analysis is GB/s of data. We must compute entropy *in situ*.
3.  **Security (Patent #5):** If the check is external, the engine can be run without it. By baking it into `InferenceControl`, the engine **cannot** generate without passing the check.

## 2. Telemetry Architecture ("The Glass Box")

We need to calculate "Cognitive ECG" stats from the raw logits *before* they are discarded, without cloning the massive tensor.

### 2.1 The `CognitiveState` Struct

This struct is computed inside `engine.rs` immediately after the forward pass and passed to the callback.

```rust
#[derive(Debug, Clone, Copy)] // Cheap to copy!
pub struct CognitiveState {
    /// How confused is the model? (Shannon entropy of softmax)
    pub entropy: f32,
    
    /// Difference between top-1 and top-2 prob. 
    /// Low gap = Hesitation/Hallucination risk.
    pub dominance_ratio: f32, 
    
    /// The raw probability of the token that was actually selected.
    pub selected_prob: f32,
}
```

### 2.2 The `AuditFrame` Struct

This is the "Receipt" (Patent #2).

```rust
pub struct AuditFrame {
    pub step_seq: u64,
    pub token_id: u32,
    pub timestamp_ns: u64,
    
    /// Hash(Previous_Hash + Self)
    pub chain_hash: [u8; 32], 
}
```

## 3. FSE Architecture ("The Semantic Radar")

Fused Semantic Execution is a `Trie`-based matcher compiled offline. At runtime, we simply walk the graph.

### 3.1 The Compiler (Offline/Startup)
-   **Input:** List of Regex/Rules (e.g., "Address", "SSN", "Shell Injection").
-   **Process:** Compile into a standard DFA (Deterministic Finite Automaton) or Aho-Corasick automaton.
-   **Output:** `FseAutomaton` (Read-only, thread-safe, `Arc`-able).

### 3.2 The Runtime (The "Walker")
-   **State:** `FseCursor`. It holds a pointer to the current node in the Automaton.
-   **Step:** `cursor = automaton.next(cursor, char)`.
-   **Cost:** $O(1)$ per character. Zero allocations.

### 3.3 Integration Point

```rust
// In src/control.rs

pub struct FseMonitor {
    automaton: Arc<FseAutomaton>,
    cursor: usize, // Current node index
    triggered_rules: Vec<RuleId>,
}

impl InferenceControl for FseMonitor {
    fn intervene(&mut self, event: InferenceEvent) -> ControlDecision {
        // 1. Update FSE Cursor with new text
        for char in event.new_text_fragment.chars() {
            self.cursor = self.automaton.next(self.cursor, char);
            
            // 2. Check for "Hot" States (Matches)
            if let Some(rule) = self.automaton.get_match(self.cursor) {
                // FAIL-CLOSED: Stop immediately
                return ControlDecision::Stop(StopReason::Safety(rule));
            }
        }
        
        // 3. Emit Telemetry (Rule Pressure)
        // (Logging code here)
        
        ControlDecision::Continue
    }
}
```

## 4. Implementation Plan

### Phase 1: The Metrics Plumbing
1.  **Modify `Engine`:** Calculate `CognitiveState` (entropy) inside the loop.
2.  **Update `InferenceEvent`:** Add `cognitive: CognitiveState` field.
3.  **Benchmark:** Ensure calculation costs < 1% of inference time.

### Phase 2: The FSE Core
1.  **Salvage:** Port `Aho-Corasick` implementation (or use the crate).
2.  **Action:** Build the `FseMonitor` that implements `InferenceControl`.
3.  **Test:** Feed "bad" strings and ensure it cuts off *mid-word*.

### Phase 3: The "Certified Invoice"
1.  **Crypto:** Add SHA-256 state tracking to the Control loop.
2.  **Output:** Write a sidecar `.audit` file alongside the output.

## 5. Artifacts
-   **New Crate?** No, keep inside `airframe` for now to access internal Types.
-   **External Dependency:** `aho-corasick` (Standard, blazingly fast).
