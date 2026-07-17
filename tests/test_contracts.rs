//! Baseline PPT contract suite for Airframe.
//!
//! Objective gate (no GPU required): these tests verify that the invariant
//! framework works AND that Airframe's core engine invariants are actually
//! exercised. Run in CI via `cargo test -p airframe --test test_contracts`.
//!
//! Each contract test:
//!   1. clears the invariant log,
//!   2. drives PRODUCTION code that embeds the relevant invariant gate
//!      (e.g. `loader::compute_chunk_plan`, `ChunkPlan::buffer_for_word`),
//!   3. asserts via [`contract_test`] that the invariant was genuinely checked.
//!
//! NOTE: the invariant log is a process-global static, so this suite MUST run
//! single-threaded (`--test-threads=1`) — see the PPT guide.

use airframe::backend::bindless::loader::{compute_chunk_plan, ChunkPlan};
use airframe::invariant_ppt::airframe_invariants::*;
use airframe::invariant_ppt::*;

#[test]
fn framework_self_test() {
    clear_invariant_log();
    assert_invariant(true, "framework alive", Some("self_test"));
    contract_test("framework_self_test", &["framework alive"]);
}

#[test]
fn storage_buffer_limit_contract() {
    clear_invariant_log();

    // Legal: well under the 2 GB binding limit.
    assert_buffer_within_limit(1024, "self_test");

    // Illegal: one byte over the limit must violate the invariant.
    let over_limit = std::panic::catch_unwind(|| {
        assert_buffer_within_limit(MAX_STORAGE_BUFFER_BINDING_SIZE + 1, "self_test");
    });
    assert!(
        over_limit.is_err(),
        "buffer over the 2 GB binding limit must violate the invariant"
    );

    contract_test("storage_buffer_limit", &[
        "Storage buffer must not exceed 2 GB binding limit",
    ]);
}

#[test]
fn alignment_contract() {
    clear_invariant_log();

    assert_alignment(256, "self_test");
    assert_alignment(16384, "self_test");

    let unaligned = std::panic::catch_unwind(|| assert_alignment(257, "self_test"));
    assert!(
        unaligned.is_err(),
        "unaligned value must violate the alignment invariant"
    );

    contract_test("alignment", &["Buffer size/offset must be 256-byte aligned"]);
}

#[test]
fn chunk_count_contract() {
    clear_invariant_log();

    assert_chunk_count_within_limit(1, "self_test");
    assert_chunk_count_within_limit(MAX_CHUNKS, "self_test");

    let too_many = std::panic::catch_unwind(|| {
        assert_chunk_count_within_limit(MAX_CHUNKS + 1, "self_test")
    });
    assert!(
        too_many.is_err(),
        "chunk count beyond MAX_CHUNKS must violate the invariant"
    );

    contract_test("chunk_count", &["Chunk count must be within [1, MAX_CHUNKS]"]);
}

#[test]
fn word_index_contract() {
    clear_invariant_log();

    let num_words: u32 = 4;
    assert_word_index_in_range(0, num_words, "self_test");
    assert_word_index_in_range(3, num_words, "self_test");

    let out_of_bounds =
        std::panic::catch_unwind(|| assert_word_index_in_range(4, num_words, "self_test"));
    assert!(
        out_of_bounds.is_err(),
        "out-of-range word index must violate the invariant"
    );

    contract_test("word_index", &["Word index must be within buffer bounds"]);
}

// ---------------------------------------------------------------------------
// Bead `a1-load-multi-buffer-core` contract + property tests.
//
// These drive the PRODUCTION `compute_chunk_plan` / `ChunkPlan::buffer_for_word`
// gates (the same functions `load_from_disk` calls), then assert via
// `contract_test` that the embedded invariants were genuinely exercised. They do
// NOT call the gates directly in the test body — the production code does.
// ---------------------------------------------------------------------------

#[test]
fn loader_chunk_plan_contract() {
    clear_invariant_log();

    // Small model (< 2 GB) → single chunk; production fn asserts align + limit.
    let plan = compute_chunk_plan(1_000_000, 2_000_000_000);
    assert_eq!(plan.num_chunks, 1, "sub-limit model must be one chunk");
    assert_eq!(
        plan.effective_chunk % REQUIRED_ALIGNMENT,
        0,
        "effective_chunk must be 256-byte aligned"
    );
    assert!(
        plan.effective_chunk <= MAX_STORAGE_BUFFER_BINDING_SIZE,
        "effective_chunk must stay within the 2 GB binding limit"
    );

    // A > 2 GB model spanning two chunks (still within MAX_CHUNKS).
    let plan2 = compute_chunk_plan(3_000_000_000, 2_000_000_000);
    assert_eq!(plan2.num_chunks, 2, "3 GB model must span two 2 GB chunks");

    // Adapter limit below 2 GB must cap (and stay aligned) — e.g. 1 GiB adapter.
    let plan3 = compute_chunk_plan(1_000_000, 1 << 30);
    assert!(plan3.effective_chunk <= (1u64 << 30));
    assert_eq!(plan3.effective_chunk % REQUIRED_ALIGNMENT, 0);

    contract_test(
        "loader_chunk_plan",
        &[
            "Buffer size/offset must be 256-byte aligned",
            "Storage buffer must not exceed 2 GB binding limit",
            "Chunk count must be within [1, MAX_CHUNKS]",
        ],
    );
}

#[test]
fn loader_chunk_plan_property() {
    // Property: for any legal file size, the plan is 256-aligned, within the
    // binding limit, and `num_chunks` covers the whole file (and only just).
    property_test("loader_chunk_plan_covers_file", || {
        let adapter_limit = 2_000_000_000u64;
        for &file_size in &[
            1u64,
            255,
            256,
            1_000_000,
            2_000_000_000,
            2_000_000_001,
            8_000_000_000,
            15_000_000_000,
        ] {
            let ChunkPlan {
                effective_chunk,
                num_chunks,
            } = compute_chunk_plan(file_size, adapter_limit);

            if effective_chunk % REQUIRED_ALIGNMENT != 0 {
                return false;
            }
            if effective_chunk > MAX_STORAGE_BUFFER_BINDING_SIZE {
                return false;
            }
            if !(1..=MAX_CHUNKS).contains(&num_chunks) {
                return false;
            }
            // num_chunks must cover the file...
            let covered = effective_chunk.saturating_mul(num_chunks as u64);
            if covered < file_size {
                return false;
            }
            // ...and not waste a whole extra buffer.
            let one_fewer = effective_chunk.saturating_mul((num_chunks - 1) as u64);
            if num_chunks > 1 && one_fewer >= file_size {
                return false;
            }
        }
        true
    });

    contract_test(
        "loader_chunk_plan_property",
        &[
            "Buffer size/offset must be 256-byte aligned",
            "Storage buffer must not exceed 2 GB binding limit",
            "Chunk count must be within [1, MAX_CHUNKS]",
        ],
    );
}

#[test]
fn loader_chunk_plan_overflow_violation() {
    clear_invariant_log();

    // A file larger than MAX_CHUNKS * 2 GB must trip the chunk-count invariant
    // (larger-model support is deferred — rejected at load, not silently split).
    let too_big = MAX_STORAGE_BUFFER_BINDING_SIZE * (MAX_CHUNKS as u64) + 1;
    let violated = std::panic::catch_unwind(|| {
        compute_chunk_plan(too_big, MAX_STORAGE_BUFFER_BINDING_SIZE);
    });
    assert!(
        violated.is_err(),
        "a model needing more than MAX_CHUNKS buffers must violate the chunk-count invariant"
    );

    contract_test(
        "loader_chunk_plan_overflow",
        &["Chunk count must be within [1, MAX_CHUNKS]"],
    );
}

#[test]
fn multi_buffer_word_resolution() {
    clear_invariant_log();

    // Under a 3-chunk plan, an absolute word index resolves to
    // (buffer_index, word_offset_in_buffer). Exercises `ChunkPlan::buffer_for_word`,
    // which embeds the word-in-range invariant.
    let plan = ChunkPlan {
        effective_chunk: 2_000_000_000,
        num_chunks: 3,
    };
    let chunk_words: u32 = (plan.effective_chunk / 4) as u32;
    let total_words: u32 = (plan.effective_chunk * plan.num_chunks as u64 / 4) as u32;

    assert_eq!(plan.buffer_for_word(0), (0, 0));
    assert_eq!(plan.buffer_for_word(chunk_words - 1), (0, chunk_words - 1));
    assert_eq!(plan.buffer_for_word(chunk_words), (1, 0));
    assert_eq!(plan.buffer_for_word(2 * chunk_words + 7), (2, 7));

    // Out-of-range word index must trip the invariant.
    let oob = std::panic::catch_unwind(|| {
        plan.buffer_for_word(total_words);
    });
    assert!(
        oob.is_err(),
        "word index at/after total_words must violate the range invariant"
    );

    contract_test(
        "multi_buffer_word_resolution",
        &["Word index must be within buffer bounds"],
    );
}
