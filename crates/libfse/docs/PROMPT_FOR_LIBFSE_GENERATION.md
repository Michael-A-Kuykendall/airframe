# GPT-5 Prompt: Project LibFSE (The "Better Crate")

**Context:**
I am building a proprietary Rust crate called `libfse` (Fused Semantic Execution) within a workspace.
The goal is to create a "Fail-Closed Policy Engine" that outperforms the standard `aho-corasick` crate for **security/compliance workloads**.

**The Core Thesis (The "Inversion"):**
The standard `aho-corasick` crate optimizes for *finding matches* and returning them to the user.
`libfse` optimizes for *executing policy* inside the hot loop.
-   **Standard:** `Find Match -> Return to User -> User Logic branches -> Resume`. (Slow, Cache Trash).
-   **LibFSE:** `Find Match -> Execute Fused Opcode -> Update Internal State -> Resume/Abort`. (Fast, Zero-Alloc).

**Your Task:**
Generate the complete source code for `libfse`.
We are wrapping the `aho-corasick` crate (using its Builder/Automaton features to handle the graph math) but replacing the *Runtime Walker*.

**Project Structure:**
-   `Cargo.toml`: dependency `aho-corasick = "1.1"`
-   `src/lib.rs`: The types.
-   `src/store.rs`: The "Opcode Table" (mapping `PatternID` -> `FseOpcode`).
-   `src/scanner.rs`: The hot loop (The "Walker").

**Specific Requirements:**

1.  **The Opcode Enum:**
    ```rust
    pub enum FseOpcode {
        Ignore,                 // Pattern found, but rule says ignore
        Record(RuleId),         // Flip a bit in the active ruleset
        Reject(RuleId),         // FAIL-CLOSED: Stop scanning immediately
        Control(ControlOp),     // context shifts (optional)
    }
    ```

2.  **The FseMap (The Store):**
    A struct that holds the `aho_corasick::Automaton` AND a dense `Vec<FseOpcode>`.
    It must provide a method `compile(rules: Vec<Rule>) -> Self`.
    
3.  **The Scanner (The Runtime):**
    -   Must accept a streaming input (or at least `&[u8]`).
    -   **CRITICAL:** It must **not** allocate `Match` objects.
    -   It acts as a `State Machine` over the byte stream.
    -   When `automaton.next_state()` indicates a match, it immediately looks up the `FseOpcode` and executes it.
    -   If `Opcode::Reject` is hit, return `Err(Violation)` instantly.

4.  **Interface:**
    ```rust
    pub struct FseScanner { ... }
    
    impl FseScanner {
        pub fn scan(&mut self, input: &[u8]) -> Result<ScanSummary, Violation>;
    }
    ```

**Tone:**
High-performance Rust. Use `dense` maps. Focus on safety and zero-cost abstractions.
Do not hallucinate the `aho-corasick` internal API—use the public `Automaton` trait (ac_automaton_dfa or similar) if exposed, or just use `AhoCorasick::find_overlapping_iter` *if* you can prove it doesn't allocate.
*Actually, better approach:* Use `AhoCorasick::try_stream_find_iter` or the low-level `lazy_dfa` if accessible. If the public API forces allocations, wrap it as tightly as possible, but ideally, we want the `Automaton` trait usage.

**Output:**
Provide the full code for `src/lib.rs`, `src/store.rs`, and `src/scanner.rs`.
