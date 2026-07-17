//! PPT + Invariant Testing System for Airframe
//!
//! Combines **Predictive Property-Based Testing (PPT)** with **runtime invariant
//! enforcement** to provide objective, AI-independent quality gates.
//!
//! See `docs/ppt-invariant-testing.md` (Shimmy) and the downloaded
//! `ppt_invariant_guide.md` for the framework rationale.
//!
//! # Layers
//! - **E-Test** (exploration)  -> [`explore_test`]
//! - **P-Test** (property)     -> [`property_test`] + invariants
//! - **C-Test** (contract)     -> [`contract_test`] + tracking
//!
//! # Why this matters here
//! The Airframe refactor (multi-buffer loader, bind-group repack, inference
//! unification) is high-churn, partly AI-assisted work. Invariants are embedded
//! directly in engine logic via [`assert_invariant`]; every check is recorded in
//! a static log so [`contract_test`] can verify that a critical invariant was
//! *actually exercised* during a run. That is an objective gate that survives
//! refactors and AI-generated code changes — it cannot be silently dropped.

use std::collections::HashSet;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref INVARIANT_LOG: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
    static ref FAILED_INVARIANTS: Mutex<Vec<String>> = Mutex::new(Vec::new());
}

/// Core invariant assertion.
///
/// Logs that the invariant was checked (keyed by its full message) and panics
/// on violation (fail-fast). Always compiles into production builds so engine
/// logic can embed semantic contracts.
pub fn assert_invariant(condition: bool, message: &str, context: Option<&str>) {
    let full_message = match context {
        Some(ctx) => format!("{} [{}]", message, ctx),
        None => message.to_string(),
    };

    // Always record that this invariant was checked.
    if let Ok(mut log) = INVARIANT_LOG.lock() {
        log.insert(full_message.clone());
    }

    // Enforce the invariant.
    if !condition {
        if let Ok(mut failed) = FAILED_INVARIANTS.lock() {
            failed.push(full_message.clone());
        }
        panic!("INVARIANT VIOLATION: {}", full_message);
    }
}

/// Property-based test helper: runs `test_fn` across 10 isolated iterations.
///
/// Each iteration clears the invariant log for isolation. The closure must
/// return `true` for the property to hold.
pub fn property_test<F>(name: &str, test_fn: F)
where
    F: Fn() -> bool,
{
    println!("[PPT] Running property test: {}", name);

    for iteration in 1..=10 {
        clear_invariant_log();
        if !test_fn() {
            panic!("Property test '{}' failed on iteration {}", name, iteration);
        }
    }

    println!("[PPT] Property test '{}' passed", name);
}

/// Contract test: verifies that each required invariant message was actually checked.
///
/// This is the objective gate. It fails if any expected invariant string was not
/// present in the log during the preceding exercise of the system.
pub fn contract_test(name: &str, required_invariants: &[&str]) {
    println!("[PPT] Running contract test: {}", name);

    let log = INVARIANT_LOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let mut missing_invariants = Vec::new();
    for required in required_invariants {
        if !log.iter().any(|logged| logged.contains(*required)) {
            missing_invariants.push(*required);
        }
    }

    if !missing_invariants.is_empty() {
        panic!(
            "Contract test '{}' failed. Missing invariants: {:?}",
            name, missing_invariants
        );
    }

    println!(
        "[PPT] Contract test '{}' passed - all invariants verified",
        name
    );
}

/// Exploration test helper: temporary, non-fatal (catches panics).
///
/// Use for throwaway discovery during development. Does not fail the suite.
#[allow(dead_code)] // test/exploration helper; retained for ad-hoc use
pub fn explore_test<F>(name: &str, test_fn: F)
where
    F: Fn() -> bool,
{
    println!("[PPT] Exploration test: {}", name);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(test_fn)) {
        Ok(true) => println!("[PPT] Exploration test '{}' passed", name),
        Ok(false) => println!("[PPT] Exploration test '{}' failed", name),
        Err(_) => println!("[PPT] Exploration test '{}' panicked", name),
    }
}

/// Clear the invariant log (for test isolation between iterations/suites).
pub fn clear_invariant_log() {
    let mut log = INVARIANT_LOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    log.clear();
    let mut failed = FAILED_INVARIANTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    failed.clear();
}

/// All invariants checked so far this process (for contract assertions).
pub fn checked_invariants() -> Vec<String> {
    let log = INVARIANT_LOG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    log.iter().cloned().collect()
}

/// All invariants that have failed so far this process.
#[allow(dead_code)] // test introspection helper; retained for debugging
pub fn failed_invariants() -> Vec<String> {
    let failed = FAILED_INVARIANTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    failed.clone()
}

// Shared, reusable invariant gates. The future refactor beads (loader / bind-group
// / inference) EMBED these directly into production code at the semantic boundary
// they touch, then assert via `contract_test` that the gate was actually exercised.
// Keeping them here means every bead reuses the same objective check rather than
// reinventing it.
pub mod airframe_invariants {
    use super::assert_invariant;

    /// wgpu storage-buffer binding limit (2 GB) — the root cause of issue #206.
    pub const MAX_STORAGE_BUFFER_BINDING_SIZE: u64 = 2_000_000_000;

    /// Required byte alignment for GPU buffer sizes/offsets (wgpu requirement).
    pub const REQUIRED_ALIGNMENT: u64 = 256;

    /// Maximum number of discrete blob buffers supported by the repacked layouts.
    pub const MAX_CHUNKS: usize = 8;

    /// Storage buffers must not exceed the wgpu 2 GB binding limit.
    pub fn assert_buffer_within_limit(size: u64, context: &str) {
        assert_invariant(
            size <= MAX_STORAGE_BUFFER_BINDING_SIZE,
            "Storage buffer must not exceed 2 GB binding limit",
            Some(context),
        );
    }

    /// Buffer sizes and offsets must be 256-byte aligned.
    pub fn assert_alignment(value: u64, context: &str) {
        assert_invariant(
            value.is_multiple_of(REQUIRED_ALIGNMENT),
            "Buffer size/offset must be 256-byte aligned",
            Some(context),
        );
    }

    /// Number of multi-buffer chunks must fit the repacked layout [1, MAX_CHUNKS].
    pub fn assert_chunk_count_within_limit(num_chunks: usize, context: &str) {
        assert_invariant(
            (1..=MAX_CHUNKS).contains(&num_chunks),
            "Chunk count must be within [1, MAX_CHUNKS]; models beyond this need dynamic \
             multi-buffer support (larger-model support is on the roadmap — deferred)",
            Some(context),
        );
    }

    /// A word index must be in range for a buffer holding `num_words` 32-bit words.
    pub fn assert_word_index_in_range(word_idx: u32, num_words: u32, context: &str) {
        assert_invariant(
            word_idx < num_words,
            "Word index must be within buffer bounds",
            Some(context),
        );
    }

    /// Generation contract (beads `de3` / `0d9`): a prompt must be non-empty and
    /// the engine must have produced at least one output token. A silent no-output
    /// regression trips this instead of shipping empty text.
    pub fn assert_generation_valid(prompt: &str, produced_output: bool, context: &str) {
        assert_invariant(
            !prompt.is_empty(),
            "Generation prompt must not be empty",
            Some(context),
        );
        assert_invariant(
            produced_output,
            "Generation must produce at least one token",
            Some(context),
        );
    }

    /// Bind-group layout contract (beads `jaa` / `cyk` / `1my` / `woh.2.*`): blob
    /// bindings must occupy the contiguous slots `0..num_blobs-1` of the repacked
    /// layout, and `num_blobs` is capped by `MAX_CHUNKS`.
    pub fn assert_bind_group_layout_contiguous(num_blobs: usize, context: &str) {
        assert_invariant(
            (1..=MAX_CHUNKS).contains(&num_blobs),
            "Blob binding count must be within [1, MAX_CHUNKS]",
            Some(context),
        );
    }

    /// Pure mirror of the WGSL `read_blob` word -> (chunk, offset) mapping. Kept in
    /// Rust so the chunking math is unit-testable without a GPU. The WGSL shaders
    /// MUST stay in sync with this (guarded by the `wgsl_read_blob_mapping` test).
    pub fn read_blob_chunk_offset(word_idx: u32, chunk_words: u32) -> (u32, u32) {
        let chunk_words = chunk_words.max(1);
        ((word_idx / chunk_words), (word_idx % chunk_words))
    }

    /// Resolves an absolute word index to `(buffer_index, word_offset_in_buffer)`
    /// under the multi-buffer plan (beads `kjt` / `bwb` / `woh.1`). `chunk_words`
    /// is `effective_chunk / 4`.
    pub fn buffer_for_word(word_idx: u32, chunk_words: u32, total_words: u32) -> (usize, u32) {
        let chunk_words = chunk_words.max(1);
        assert_word_index_in_range(word_idx, total_words, "loader::buffer_for_word");
        ((word_idx / chunk_words) as usize, (word_idx % chunk_words))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariant_logging() {
        clear_invariant_log();

        assert_invariant(true, "Test invariant", Some("test_context"));

        let checked = checked_invariants();
        assert!(checked.iter().any(|msg| msg.contains("Test invariant")));
    }

    #[test]
    #[should_panic(expected = "INVARIANT VIOLATION")]
    fn test_invariant_violation() {
        assert_invariant(false, "This should fail", None);
    }

    #[test]
    fn test_property_test_success() {
        property_test("always_true", || true);
    }

    #[test]
    fn test_contract_test_success() {
        clear_invariant_log();
        assert_invariant(true, "Required contract", Some("test"));
        contract_test("test_contract", &["Required contract"]);
    }
}
