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

use airframe::backend::bindless::loader::{compute_chunk_plan, BlobWindow, ChunkPlan};
use airframe::backend::bindless::pipeline::LayerOffsets;
use airframe::invariant_ppt::airframe_invariants::*;
use airframe::invariant_ppt::*;

/// 128 MiB adapter limit used by the Ubuntu D3D12 path on RTX 3060.
const ADAPTER_LIMIT_128_MIB: u64 = 128 * 1024 * 1024;

/// Words per chunk at 128 MiB.
const CHUNK_WORDS_128_MIB: u32 = (ADAPTER_LIMIT_128_MIB / 4) as u32;

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

    contract_test(
        "storage_buffer_limit",
        &["Storage buffer must not exceed 2 GB binding limit"],
    );
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

    contract_test(
        "alignment",
        &["Buffer size/offset must be 256-byte aligned"],
    );
}

#[test]
fn chunk_count_contract() {
    clear_invariant_log();

    assert_chunk_count_within_limit(1, "self_test");
    assert_chunk_count_within_limit(MAX_CHUNKS, "self_test");

    let too_many =
        std::panic::catch_unwind(|| assert_chunk_count_within_limit(MAX_CHUNKS + 1, "self_test"));
    assert!(
        too_many.is_err(),
        "chunk count beyond MAX_CHUNKS must violate the invariant"
    );

    contract_test(
        "chunk_count",
        &["Chunk count must be within [1, MAX_CHUNKS]"],
    );
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

    // A > 2 GB model spanning two chunks.
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
            // num_chunks can be any positive number (no MAX_CHUNKS cap for resident chunks)
            if num_chunks == 0 {
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
        ],
    );
}

#[test]
fn loader_chunk_plan_overflow_violation() {
    clear_invariant_log();

    // The old MAX_CHUNKS cap on resident chunks is removed.
    // A 20 GB model now produces 10 resident chunks (at 2 GB/chunk) without error.
    let plan = compute_chunk_plan(20_000_000_000, MAX_STORAGE_BUFFER_BINDING_SIZE);
    assert_eq!(plan.num_chunks, 10);
    assert_eq!(plan.effective_chunk, MAX_STORAGE_BUFFER_BINDING_SIZE);

    // The limit is now enforced at dispatch time via BlobWindow span gate:
    // a window wider than BLOB_BINDING_SLOTS (8) fails before dispatch.
    let too_wide = BlobWindow::new(0, 9, 500_000_000, 10);
    assert!(too_wide.is_err(), "9-slot window must fail");

    // Also verify that a model beyond 8 GiB packed-offset limit would be rejected
    // by pack_blob_offset (but that's a separate test).

    contract_test(
        "loader_chunk_plan_overflow",
        &[], // No invariants directly exercised in this test (BlobWindow tested separately)
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

// ---------------------------------------------------------------------------
// Bead `airframe-2z6` — Llama-3.2-3B argmax regression: window algebra tests.
// These verify the sliding eight-slot BlobWindow abstraction over 16 resident
// chunks under a 128 MiB adapter binding limit (Ubuntu D3D12 on RTX 3060).
// ---------------------------------------------------------------------------

#[test]
fn loader_16_resident_chunks_plan() {
    clear_invariant_log();

    // Llama-3.2-3B Q4_K_M is ~1.9 GiB. At 128 MiB/chunk → 16 resident chunks.
    // Use a file size that's a multiple of 4 and yields 16 chunks.
    // 16 chunks at 128 MiB: file_size in (15*128MiB, 16*128MiB] = (2,013,265,920, 2,147,483,648]
    // Use 2,040,109,464 (1.9 GiB rounded to multiple of 4).
    let file_size = 2_040_109_464u64; // ~1.9 GiB, multiple of 4
    let plan = compute_chunk_plan(file_size, ADAPTER_LIMIT_128_MIB);

    // Plan must produce 16 resident chunks without violating the binding limit.
    assert_eq!(
        plan.num_chunks, 16,
        "1.9 GiB / 128 MiB = 16 resident chunks"
    );
    assert_eq!(plan.effective_chunk, ADAPTER_LIMIT_128_MIB);
    assert_eq!(plan.effective_chunk % REQUIRED_ALIGNMENT, 0);
    assert!(plan.effective_chunk <= MAX_STORAGE_BUFFER_BINDING_SIZE);

    // Each allocation stays within the adapter limit.
    for i in 0..plan.num_chunks {
        let offset = (i as u64) * plan.effective_chunk;
        let chunk_size = (file_size - offset).min(plan.effective_chunk);
        assert!(chunk_size <= ADAPTER_LIMIT_128_MIB);
        assert!(chunk_size.is_multiple_of(4));
    }

    // Final chunk covered exactly.
    let covered = plan.effective_chunk * plan.num_chunks as u64;
    assert!(covered >= file_size);
    let one_fewer = plan.effective_chunk * (plan.num_chunks - 1) as u64;
    assert!(one_fewer < file_size);

    contract_test(
        "loader_16_resident_chunks",
        &[
            "Buffer size/offset must be 256-byte aligned",
            "Storage buffer must not exceed 2 GB binding limit",
        ],
    );
}

#[test]
fn blob_window_algebra_start_zero() {
    clear_invariant_log();

    // Window starting at chunk 0 (first 8 slots).
    let window = BlobWindow::new(0, 8, CHUNK_WORDS_128_MIB, 16).expect("valid window");

    // Base word is 0.
    assert_eq!(window.window_base_words(), 0);

    // Absolute 0 → slot 0, offset 0.
    assert_eq!(window.absolute_to_local(0).unwrap(), (0, 0));

    // Last word of slot 0.
    assert_eq!(
        window.absolute_to_local(CHUNK_WORDS_128_MIB - 1).unwrap(),
        (0, CHUNK_WORDS_128_MIB - 1)
    );

    // First word of slot 1.
    assert_eq!(
        window.absolute_to_local(CHUNK_WORDS_128_MIB).unwrap(),
        (1, 0)
    );

    // Last word of slot 7 (window end - 1).
    let last_word = 8 * CHUNK_WORDS_128_MIB - 1;
    assert_eq!(
        window.absolute_to_local(last_word).unwrap(),
        (7, CHUNK_WORDS_128_MIB - 1)
    );

    // Inverse reconstruction matches.
    for slot in 0..8 {
        for &offset in &[
            0u32,
            1,
            100,
            CHUNK_WORDS_128_MIB / 2,
            CHUNK_WORDS_128_MIB - 1,
        ] {
            let abs = window.local_to_absolute(slot, offset);
            let (s, o) = window.absolute_to_local(abs).unwrap();
            assert_eq!(s, slot, "slot mismatch for abs={abs}");
            assert_eq!(o, offset, "offset mismatch for abs={abs}");
        }
    }

    // Word before window (underflow) → error.
    let _under = window.absolute_to_local(0).unwrap(); // 0 is in window
                                                       // Actually test a word that would underflow if base > 0
                                                       // For start=0, base=0, so no underflow possible. Test contains instead.
    assert!(window.contains(0));
    assert!(window.contains(last_word));
    assert!(!window.contains(last_word + 1));

    contract_test(
        "blob_window_algebra_start_zero",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn blob_window_algebra_start_nonzero() {
    clear_invariant_log();

    // Window starting at chunk 8 (slots 8..15 of 16 resident chunks).
    let window = BlobWindow::new(8, 8, CHUNK_WORDS_128_MIB, 16).expect("valid window");

    // Base word = 8 * chunk_words.
    let base = 8 * CHUNK_WORDS_128_MIB;
    assert_eq!(window.window_base_words(), base);

    // First word of window (absolute = base) → slot 0, offset 0.
    assert_eq!(window.absolute_to_local(base).unwrap(), (0, 0));

    // Last word of window (absolute = base + 8*chunk_words - 1) → slot 7, offset chunk_words-1.
    let last = base + 8 * CHUNK_WORDS_128_MIB - 1;
    assert_eq!(
        window.absolute_to_local(last).unwrap(),
        (7, CHUNK_WORDS_128_MIB - 1)
    );

    // Word before window → error.
    let before = window.absolute_to_local(base - 1);
    assert!(
        before.is_err(),
        "word before window must error: {:?}",
        before
    );

    // Word at window end → error.
    let at_end = window.absolute_to_local(base + 8 * CHUNK_WORDS_128_MIB);
    assert!(
        at_end.is_err(),
        "word at window end must error: {:?}",
        at_end
    );

    // Word after window → error.
    let after = window.absolute_to_local(base + 8 * CHUNK_WORDS_128_MIB + 100);
    assert!(after.is_err(), "word after window must error: {:?}", after);

    // Inverse reconstruction matches.
    for slot in 0..8 {
        for &offset in &[
            0u32,
            1,
            100,
            CHUNK_WORDS_128_MIB / 2,
            CHUNK_WORDS_128_MIB - 1,
        ] {
            let abs = window.local_to_absolute(slot, offset);
            let (s, o) = window.absolute_to_local(abs).unwrap();
            assert_eq!(s, slot, "slot mismatch for abs={abs}");
            assert_eq!(o, offset, "offset mismatch for abs={abs}");
        }
    }

    contract_test(
        "blob_window_algebra_start_nonzero",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn blob_window_span_gate() {
    clear_invariant_log();

    // Exactly 8 slots → passes.
    let ok = BlobWindow::new(0, 8, CHUNK_WORDS_128_MIB, 16);
    assert!(ok.is_ok(), "8-slot window must succeed: {:?}", ok);

    // 9 slots → fails before dispatch.
    let nine = BlobWindow::new(0, 9, CHUNK_WORDS_128_MIB, 16);
    assert!(nine.is_err(), "9-slot window must fail: {:?}", nine);

    // Window extending past resident chunks → fails.
    let past = BlobWindow::new(10, 8, CHUNK_WORDS_128_MIB, 16); // 10+8=18 > 16
    assert!(
        past.is_err(),
        "window past resident chunks must fail: {:?}",
        past
    );

    // Start chunk >= total → fails.
    let bad_start = BlobWindow::new(16, 8, CHUNK_WORDS_128_MIB, 16);
    assert!(
        bad_start.is_err(),
        "start >= total must fail: {:?}",
        bad_start
    );

    // Slot count 0 → fails.
    let zero_slots = BlobWindow::new(0, 0, CHUNK_WORDS_128_MIB, 16);
    assert!(
        zero_slots.is_err(),
        "zero slots must fail: {:?}",
        zero_slots
    );

    // Slot count > BLOB_BINDING_SLOTS → fails.
    let too_many = BlobWindow::new(0, 9, CHUNK_WORDS_128_MIB, 16);
    assert!(
        too_many.is_err(),
        "slots > BLOB_BINDING_SLOTS must fail: {:?}",
        too_many
    );

    contract_test(
        "blob_window_span_gate",
        &[], // Constructor validation, not word-index invariant
    );
}

#[test]
fn blob_window_binding_resources() {
    clear_invariant_log();

    // This test would require a real BindlessModel, which needs a GPU.
    // Instead, we test the logic by checking the window construction and
    // the binding_resources method signature exists. The actual GPU binding
    // is tested in integration tests with a real device.
    let window = BlobWindow::new(0, 8, CHUNK_WORDS_128_MIB, 16).unwrap();

    // Verify window properties for binding.
    assert_eq!(window.start_chunk, 0);
    assert_eq!(window.slot_count, 8);
    assert_eq!(window.chunk_words, CHUNK_WORDS_128_MIB);
    assert_eq!(window.total_resident_chunks, 16);
    assert_eq!(window.window_base_words(), 0);

    contract_test(
        "blob_window_binding_resources",
        &[], // No invariants directly exercised in this unit test
    );
}

#[test]
fn sentinel_gate_nonzero_packed_offsets() {
    clear_invariant_log();

    // This test verifies the metadata.rs fix: present minimum-offset tensors
    // never encode as packed offset 0 (which is reserved for "missing").
    // The test in metadata.rs already covers this, but we add a contract test
    // here to ensure the invariant is exercised.

    // The fix: base_byte = (min_offset & !3).saturating_sub(4)
    // So the minimum tensor offset gets relative_packed_offset = (min - base) / 2 = 2 (nonzero).
    use airframe::backend::bindless::metadata::BindlessMetadata;
    use std::collections::HashMap;

    let mut tensor_offsets = HashMap::new();
    // ffn_gate at offset 100 (minimum)
    tensor_offsets.insert("blk.20.ffn_gate.weight".to_string(), 100u64);
    tensor_offsets.insert("blk.20.attn_norm.weight".to_string(), 200u64);
    // Required tensors for get_layer_offsets
    tensor_offsets.insert("blk.20.attn_q.weight".to_string(), 300u64);
    tensor_offsets.insert("blk.20.attn_k.weight".to_string(), 400u64);
    tensor_offsets.insert("blk.20.attn_v.weight".to_string(), 500u64);
    tensor_offsets.insert("blk.20.attn_output.weight".to_string(), 600u64);
    tensor_offsets.insert("blk.20.ffn_norm.weight".to_string(), 700u64);
    tensor_offsets.insert("blk.20.ffn_down.weight".to_string(), 800u64);
    tensor_offsets.insert("blk.20.ffn_up.weight".to_string(), 900u64);

    let mut tensor_types = HashMap::new();
    tensor_types.insert("blk.20.ffn_gate.weight".to_string(), 12); // Q4_K
    tensor_types.insert("blk.20.attn_norm.weight".to_string(), 12);
    tensor_types.insert("blk.20.attn_q.weight".to_string(), 12);
    tensor_types.insert("blk.20.attn_k.weight".to_string(), 12);
    tensor_types.insert("blk.20.attn_v.weight".to_string(), 12);
    tensor_types.insert("blk.20.attn_output.weight".to_string(), 12);
    tensor_types.insert("blk.20.ffn_norm.weight".to_string(), 12);
    tensor_types.insert("blk.20.ffn_down.weight".to_string(), 12);
    tensor_types.insert("blk.20.ffn_up.weight".to_string(), 12);

    let meta = BindlessMetadata {
        version: 3,
        tensor_count: 9,
        tensor_offsets,
        tensor_types,
        tensor_dims: HashMap::new(),
        data_start_offset: 0,
        gguf_metadata: HashMap::new(),
        compiled_layers: vec![],
    };

    let offs = meta
        .get_layer_offsets(20, "llama")
        .expect("layer 20 exists");

    // ffn_gate is the minimum tensor → its packed offset must be NONZERO.
    assert_ne!(
        offs.ffn_gate, 0,
        "minimum tensor must not encode as zero (sentinel)"
    );

    // attn_norm should have a different (larger) packed offset.
    assert_ne!(offs.attn_norm, 0);
    assert_ne!(offs.attn_norm, offs.ffn_gate);

    contract_test(
        "sentinel_gate_nonzero_packed_offsets",
        &[], // The actual invariant is in relative_packed_offset / metadata.rs
    );
}

#[test]
fn layer20_fixture_ffn_gate_reconstructs() {
    clear_invariant_log();

    // Layer-20 fixture: ffn_gate is the minimum tensor and reconstructs to its
    // original GGUF address through a nonzero packed offset and nonzero window start.
    use airframe::backend::bindless::metadata::BindlessMetadata;
    use airframe::backend::bindless::pipeline::relative_packed_offset;
    use std::collections::HashMap;

    // Simulate Llama-3.2-3B layer 20 where ffn_gate.weight is the minimum offset tensor.
    // Absolute offsets (from GGUF):
    let ffn_gate_abs = 1_500_000_000u64; // ~1.5 GiB - minimum in layer
    let attn_norm_abs = ffn_gate_abs + 100_000;
    let ffn_up_abs = ffn_gate_abs + 200_000;
    let ffn_down_abs = ffn_gate_abs + 300_000;
    // Required tensors for get_layer_offsets
    let attn_q_abs = ffn_gate_abs + 400_000;
    let attn_k_abs = ffn_gate_abs + 500_000;
    let attn_v_abs = ffn_gate_abs + 600_000;
    let attn_output_abs = ffn_gate_abs + 700_000;
    let ffn_norm_abs = ffn_gate_abs + 800_000;

    let mut tensor_offsets = HashMap::new();
    tensor_offsets.insert("blk.20.ffn_gate.weight".to_string(), ffn_gate_abs);
    tensor_offsets.insert("blk.20.attn_norm.weight".to_string(), attn_norm_abs);
    tensor_offsets.insert("blk.20.ffn_up.weight".to_string(), ffn_up_abs);
    tensor_offsets.insert("blk.20.ffn_down.weight".to_string(), ffn_down_abs);
    tensor_offsets.insert("blk.20.attn_q.weight".to_string(), attn_q_abs);
    tensor_offsets.insert("blk.20.attn_k.weight".to_string(), attn_k_abs);
    tensor_offsets.insert("blk.20.attn_v.weight".to_string(), attn_v_abs);
    tensor_offsets.insert("blk.20.attn_output.weight".to_string(), attn_output_abs);
    tensor_offsets.insert("blk.20.ffn_norm.weight".to_string(), ffn_norm_abs);

    let mut tensor_types = HashMap::new();
    for k in tensor_offsets.keys() {
        tensor_types.insert(k.clone(), 12); // Q4_K
    }

    let meta = BindlessMetadata {
        version: 3,
        tensor_count: 9,
        tensor_offsets,
        tensor_types,
        tensor_dims: HashMap::new(),
        data_start_offset: 0,
        gguf_metadata: HashMap::new(),
        compiled_layers: vec![],
    };

    let offs = meta
        .get_layer_offsets(20, "llama")
        .expect("layer 20 exists");

    // ffn_gate is minimum → base is one aligned word before it.
    let min_offset = ffn_gate_abs;
    let base_byte = (min_offset & !3u64).saturating_sub(4);
    let blob_base_words = (base_byte / 4) as u32;

    // Packed offset for ffn_gate = (ffn_gate_abs - base_byte) / 2 = 2 (nonzero).
    let packed = relative_packed_offset(ffn_gate_abs, base_byte).expect("valid offset");
    assert_eq!(packed, 2);
    assert_eq!(offs.ffn_gate, packed);

    // Reconstruct absolute word index: gow(pack) = pack / 2 + blob_base_words
    let reconstructed_word = (packed / 2) + blob_base_words;
    let original_word = (ffn_gate_abs / 4) as u32;
    assert_eq!(
        reconstructed_word, original_word,
        "reconstructed word {} != original word {}",
        reconstructed_word, original_word
    );

    // Now test with a nonzero window start (simulating window starting at chunk 8).
    let chunk_words = CHUNK_WORDS_128_MIB;
    let window = BlobWindow::new(8, 8, chunk_words, 16).unwrap();
    let window_base = window.window_base_words();

    // The shader sees local_word = original_word - window_base
    // But blob_base_words in LayerParams is window-local: blob_base_words - window_base
    let layer_blob_base_words = blob_base_words - window_base;

    // Reconstruct through window: local_word = pack/2 + layer_blob_base_words
    // absolute = window_base + local_word
    let local_reconstructed = (packed / 2) + layer_blob_base_words;
    let absolute_reconstructed = window_base + local_reconstructed;
    assert_eq!(absolute_reconstructed, original_word);

    contract_test("layer20_fixture_ffn_gate_reconstructs", &[]);
}

// ---------------------------------------------------------------------------
// Bead `airframe-vws` — window algebra for the split layer pipeline entry
// points (`run_qkv_only_test`, `run_layer_stepwise_test`, `run_layer_with_cache`,
// `run_layer_with_cache_int4`, `run_layer_with_cache_debug`).
//
// Every one of those dispatches now routes its blob bindings and its
// `blob_base_words` through `plan_layer_window`. These tests pin that algebra
// on the CPU: window selection, window-local rebasing, and round-trip
// reconstruction of each tensor's absolute word index.
// ---------------------------------------------------------------------------

/// Builds a `LayerOffsets` whose tensors are laid out at increasing packed
/// offsets from `base_byte`, mirroring the GGUF layout for one Llama block.
fn layer_offsets_fixture(base_byte: u64, tensor_abs: &[u64]) -> (LayerOffsets, u32) {
    use airframe::backend::bindless::pipeline::relative_packed_offset;

    let mut offs = LayerOffsets {
        attn_norm: 0,
        attn_norm_bias: 0,
        attn_q: 0,
        attn_k: 0,
        attn_v: 0,
        attn_out: 0,
        ffn_norm: 0,
        ffn_norm_bias: 0,
        ffn_gate: 0,
        ffn_down: 0,
        ffn_up: 0,
        layer_idx: 20,
        attn_q_norm: 0,
        attn_k_norm: 0,
        attn_q_bias: 0,
        attn_k_bias: 0,
        attn_v_bias: 0,
        v_is_q4k: 0,
        ffn_down_is_q4k: 0,
    };

    let packed: Vec<u32> = tensor_abs
        .iter()
        .map(|&abs| relative_packed_offset(abs, base_byte).expect("valid packed offset"))
        .collect();

    offs.attn_norm = packed[0];
    offs.attn_q = packed[1];
    offs.attn_k = packed[2];
    offs.attn_v = packed[3];
    offs.attn_out = packed[4];
    offs.ffn_norm = packed[5];
    offs.ffn_gate = packed[6];
    offs.ffn_up = packed[7];
    offs.ffn_down = packed[8];

    (offs, (base_byte / 4) as u32)
}

#[test]
fn layer_window_plan_rebases_blob_base_words() {
    clear_invariant_log();

    use airframe::backend::bindless::pipeline::plan_layer_window;

    // Llama-3.2-3B layer 20: tensors sit ~1.5 GiB into the file, which at
    // 128 MiB/chunk is resident chunk 11 — well past slot 0. This is exactly
    // the case that regressed: a window starting at chunk 0 cannot reach it.
    let base_byte = 1_500_000_000u64 & !3u64;
    let tensor_abs: Vec<u64> = (0..9).map(|i| base_byte + 4 + i * 100_000).collect();
    let (offs, blob_base_words) = layer_offsets_fixture(base_byte, &tensor_abs);

    let (window, local_base) =
        plan_layer_window(&offs, blob_base_words, CHUNK_WORDS_128_MIB, 16).expect("plan succeeds");
    let window = window.expect("layer declares tensors → window must be Some");

    // The window must start at the chunk actually holding the tensors, not 0.
    let expected_start = (blob_base_words / CHUNK_WORDS_128_MIB) as usize;
    assert_eq!(
        window.start_chunk, expected_start,
        "window must start at the resident chunk holding the layer"
    );
    assert!(
        window.start_chunk > 0,
        "fixture must exercise a nonzero start"
    );
    assert!(
        window.slot_count <= 8,
        "window must fit the shader's 8 slots"
    );

    // blob_base_words handed to the shader is window-local.
    assert_eq!(
        local_base,
        blob_base_words - window.window_base_words(),
        "blob_base_words must be rebased to the window start"
    );

    // Round-trip: every tensor reconstructs to its original absolute word via
    // the shader's own arithmetic — gow(pack) = pack/2 + blob_base_words —
    // then back out through the window.
    for (&abs, &packed) in tensor_abs.iter().zip(
        [
            offs.attn_norm,
            offs.attn_q,
            offs.attn_k,
            offs.attn_v,
            offs.attn_out,
            offs.ffn_norm,
            offs.ffn_gate,
            offs.ffn_up,
            offs.ffn_down,
        ]
        .iter(),
    ) {
        let local_word = (packed / 2) + local_base;
        let (slot, offset) = window
            .absolute_to_local(window.window_base_words() + local_word)
            .expect("tensor word must be inside the bound window");
        assert!(slot < window.slot_count, "slot must be bound");
        assert_eq!(
            window.local_to_absolute(slot, offset),
            (abs / 4) as u32,
            "tensor at {} must reconstruct to its original word",
            abs
        );
    }

    contract_test(
        "layer_window_plan_rebases_blob_base_words",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn layer_window_plan_start_chunk_zero_is_identity() {
    clear_invariant_log();

    use airframe::backend::bindless::pipeline::plan_layer_window;

    // Layer 0 lives in chunk 0. The window must degenerate to the identity
    // mapping so single-chunk models (and every early layer) are unchanged.
    let base_byte = 4_096u64;
    let tensor_abs: Vec<u64> = (0..9).map(|i| base_byte + 4 + i * 1_000).collect();
    let (offs, blob_base_words) = layer_offsets_fixture(base_byte, &tensor_abs);

    let (window, local_base) =
        plan_layer_window(&offs, blob_base_words, CHUNK_WORDS_128_MIB, 16).expect("plan succeeds");
    let window = window.expect("layer declares tensors → window must be Some");

    assert_eq!(window.start_chunk, 0);
    assert_eq!(window.window_base_words(), 0);
    assert_eq!(
        local_base, blob_base_words,
        "a chunk-0 window must not rebase blob_base_words"
    );

    contract_test("layer_window_plan_start_chunk_zero_is_identity", &[]);
}

#[test]
fn layer_window_plan_rejects_oversized_span() {
    clear_invariant_log();

    use airframe::backend::bindless::pipeline::plan_layer_window;

    // A layer whose tensors straddle more than BLOB_BINDING_SLOTS chunks
    // cannot be bound in one dispatch. That must be a hard error, never a
    // silent read from an unbound slot.
    let base_byte = 0u64;
    let span = (CHUNK_WORDS_128_MIB as u64) * 4; // one chunk in bytes
    let tensor_abs: Vec<u64> = (0..9).map(|i| 4 + i * span).collect();
    let (offs, blob_base_words) = layer_offsets_fixture(base_byte, &tensor_abs);

    let result = plan_layer_window(&offs, blob_base_words, CHUNK_WORDS_128_MIB, 16);
    assert!(
        result.is_err(),
        "a span wider than 8 chunks must be rejected: {:?}",
        result
    );

    contract_test("layer_window_plan_rejects_oversized_span", &[]);
}

#[test]
fn layer_window_plan_no_tensors_is_noop() {
    clear_invariant_log();

    use airframe::backend::bindless::pipeline::plan_layer_window;

    // A layer with no declared tensors has no span to cover; the planner must
    // pass blob_base_words through untouched rather than fabricate a window.
    let (mut offs, _) = layer_offsets_fixture(0, &[4; 9]);
    offs.attn_norm = 0;
    offs.attn_q = 0;
    offs.attn_k = 0;
    offs.attn_v = 0;
    offs.attn_out = 0;
    offs.ffn_norm = 0;
    offs.ffn_gate = 0;
    offs.ffn_up = 0;
    offs.ffn_down = 0;

    let (window, local_base) =
        plan_layer_window(&offs, 12_345, CHUNK_WORDS_128_MIB, 16).expect("plan succeeds");

    assert!(window.is_none(), "no tensors → no window");
    assert_eq!(local_base, 12_345, "blob_base_words must pass through");

    contract_test("layer_window_plan_no_tensors_is_noop", &[]);
}

// ---------------------------------------------------------------------------
// Bead `airframe-8r3` — RMSNorm window algebra.
//
// sh_rmsnorm.wgsl has NO blob_base_words uniform: it indexes read_blob with
// `weight_offset + i` and `bias_offset + i` directly. So for a norm tensor
// living past resident chunk 7, BOTH offsets must be rebased onto the bound
// window, and the window must span the tensor's whole `count`-element run —
// not merely the word it starts at.
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_window_covers_full_count_span() {
    clear_invariant_log();

    // A norm tensor starting near the end of a chunk whose `count` elements
    // spill into the next chunk. A window derived from the start word alone
    // would be one slot short and the tail would read the wrong buffer.
    let count = 4_096u32;
    let weight_offset = CHUNK_WORDS_128_MIB - 16; // 16 words before the boundary
    let end_word = weight_offset + count - 1;

    assert_eq!(
        weight_offset / CHUNK_WORDS_128_MIB,
        0,
        "fixture must start in chunk 0"
    );
    assert_eq!(
        end_word / CHUNK_WORDS_128_MIB,
        1,
        "fixture must spill into chunk 1"
    );

    let window = BlobWindow::for_range(weight_offset, end_word, CHUNK_WORDS_128_MIB, 16)
        .expect("valid window");

    assert_eq!(window.start_chunk, 0);
    assert_eq!(
        window.slot_count, 2,
        "window must span both chunks the tensor touches"
    );

    // Every element of the run resolves inside the bound window.
    for i in [0u32, 1, 15, 16, count / 2, count - 1] {
        let (slot, _) = window
            .absolute_to_local(weight_offset + i)
            .unwrap_or_else(|e| panic!("element {} outside window: {}", i, e));
        assert!(slot < window.slot_count, "element {} in unbound slot", i);
    }

    contract_test(
        "rmsnorm_window_covers_full_count_span",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn rmsnorm_offsets_rebase_onto_window() {
    clear_invariant_log();

    // output_norm.weight for Llama-3.2-3B sits ~1.9 GiB in — resident chunk 14
    // at 128 MiB. The shader's absolute-index reads only land correctly if the
    // host subtracts the window base from weights_offset/bias_offset.
    let count = 3_072u32; // n_embd for Llama-3.2-3B
    let weight_offset = 14 * CHUNK_WORDS_128_MIB + 1_000;
    let bias_offset = weight_offset + count;

    let window = BlobWindow::for_range(
        weight_offset,
        bias_offset + count - 1,
        CHUNK_WORDS_128_MIB,
        16,
    )
    .expect("valid window");
    let window_base = window.window_base_words();

    assert_eq!(window.start_chunk, 14);
    assert!(
        window_base > 0,
        "fixture must exercise a nonzero window base"
    );

    let local_weight = weight_offset - window_base;
    let local_bias = bias_offset - window_base;

    // Rebased offsets must be reachable through the window's own slots, and
    // must round-trip back to the original absolute words.
    for (local, absolute, label) in [
        (local_weight, weight_offset, "weight"),
        (local_bias, bias_offset, "bias"),
    ] {
        for i in [0u32, 1, count / 2, count - 1] {
            let (slot, offset) = window
                .absolute_to_local(window_base + local + i)
                .unwrap_or_else(|e| panic!("{} element {} outside window: {}", label, i, e));
            assert_eq!(
                window.local_to_absolute(slot, offset),
                absolute + i,
                "{} element {} must round-trip to its absolute word",
                label,
                i
            );
        }
    }

    // A real bias must never rebase onto the shader's disabled sentinel.
    assert_ne!(
        local_bias, 0,
        "rebased bias must not collide with the bias-disabled sentinel"
    );

    contract_test(
        "rmsnorm_offsets_rebase_onto_window",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn rmsnorm_window_chunk_zero_is_identity() {
    clear_invariant_log();

    // Early/small models keep the norm in chunk 0; rebasing must be a no-op so
    // the single-chunk path is bit-identical to before the window abstraction.
    let count = 2_048u32;
    let weight_offset = 4_096u32;

    let window = BlobWindow::for_range(
        weight_offset,
        weight_offset + count - 1,
        CHUNK_WORDS_128_MIB,
        16,
    )
    .expect("valid window");

    assert_eq!(window.start_chunk, 0);
    assert_eq!(window.window_base_words(), 0);
    assert_eq!(
        weight_offset - window.window_base_words(),
        weight_offset,
        "a chunk-0 window must not rebase the weight offset"
    );

    contract_test("rmsnorm_window_chunk_zero_is_identity", &[]);
}

// ---------------------------------------------------------------------------
// Bead `airframe-djk` — prefill/decode + LM-head window algebra.
//
// sh_head_blob.wgsl reads `weight_off + rel_word` with no separate base
// uniform, so weight_off must be rebased onto the bound window. For a tile
// whose rows start past the window base, that rebase is negative and relies on
// u32 wraparound — which these tests pin explicitly.
//
// The head row stride also must come from the GGUF spec registry
// (block_bytes / block_elems), not a 4-bytes-per-element assumption: a Q4_K
// head is ~4.5 bits/element, so f32 math overstates the span ~7x.
// ---------------------------------------------------------------------------

/// Row stride in bytes for a quantized LM head row, per the GGUF spec registry.
fn head_row_bytes(dim: u64, block_elems: u64, block_bytes: u64) -> u64 {
    assert!(
        dim.is_multiple_of(block_elems),
        "dim must fill whole blocks"
    );
    (dim / block_elems) * block_bytes
}

#[test]
fn lm_head_row_stride_follows_quant_geometry() {
    clear_invariant_log();

    // Llama-3.2-3B: n_embd = 3072, n_vocab = 128256, Q6_K head.
    // Q6_K = 256 elems / 210 bytes (registry). A 4-byte/element assumption
    // would claim 12288 B/row against the true 2520 B/row — ~4.9x too large,
    // which is exactly how a bindable span becomes unbindable.
    let dim = 3_072u64;
    let q6k = head_row_bytes(dim, 256, 210);
    assert_eq!(q6k, 2_520, "Q6_K row = (3072/256)*210");

    let naive_f32 = dim * 4;
    assert!(
        naive_f32 > q6k * 4,
        "f32 assumption must be shown to grossly overstate the span"
    );

    // Q4_K = 256 elems / 144 bytes.
    assert_eq!(head_row_bytes(dim, 256, 144), 1_728);
    // F16 = 1 elem / 2 bytes.
    assert_eq!(head_row_bytes(dim, 1, 2), 6_144);

    contract_test("lm_head_row_stride_follows_quant_geometry", &[]);
}

#[test]
fn lm_head_full_vocab_window_is_bindable() {
    clear_invariant_log();

    // Full-vocab head dispatch for Llama-3.2-3B Q6_K, weights near the end of
    // the file (~1.6 GiB in). The whole head must fit within 8 bound slots.
    let dim = 3_072u64;
    let vocab = 128_256u64;
    let row_bytes = head_row_bytes(dim, 256, 210);
    let weight_byte = 1_600_000_000u64;

    let start_word = (weight_byte / 4) as u32;
    let end_word = ((weight_byte + vocab * row_bytes - 1) / 4) as u32;

    let window = BlobWindow::for_range(start_word, end_word, CHUNK_WORDS_128_MIB, 16)
        .expect("full-vocab Q6_K head must be bindable in 8 slots");

    assert!(window.slot_count <= 8);

    // Drive the real resolution path (not just `contains`) so the word-range
    // invariant is genuinely exercised for both ends of the head.
    for (word, label) in [(start_word, "start"), (end_word, "end")] {
        let (slot, offset) = window
            .absolute_to_local(word)
            .unwrap_or_else(|e| panic!("head {} outside window: {}", label, e));
        assert!(slot < window.slot_count, "head {} in unbound slot", label);
        assert_eq!(window.local_to_absolute(slot, offset), word);
    }

    contract_test(
        "lm_head_full_vocab_window_is_bindable",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn lm_head_tile_weight_off_rebase_wraps_correctly() {
    clear_invariant_log();

    use airframe::backend::bindless::pipeline::rebase_head_weight_off;

    // A late tile: its rows start far past the weight tensor start, so the
    // tile window base EXCEEDS weight_off and the rebase goes negative.
    // WGSL u32 addition wraps, so wrapped_base + rel_word must still land on
    // the correct window-local word for every row the tile reads.
    let dim = 3_072u64;
    let row_bytes = head_row_bytes(dim, 256, 210);
    let weight_byte = 1_000_000u64; // head starts early in the file
    let weight_off = (weight_byte / 4) as u32;

    let base_row = 100_000u64; // deep into the vocab
    let rows = 4_096u64;

    let tile_start_word = ((weight_byte + base_row * row_bytes) / 4) as u32;
    let tile_end_word = ((weight_byte + (base_row + rows) * row_bytes - 1) / 4) as u32;

    let window = BlobWindow::for_range(tile_start_word, tile_end_word, CHUNK_WORDS_128_MIB, 16)
        .expect("tile must be bindable");

    assert!(
        window.window_base_words() > weight_off,
        "fixture must exercise the negative-rebase case"
    );

    let wrapped = rebase_head_weight_off(weight_off, &window);

    // For every rel_word the tile actually reads, the shader's
    // `wrapped + rel_word` (u32, wrapping) must equal the true window-local
    // word, and resolve to a bound slot.
    let first_rel = (base_row * row_bytes / 4) as u32;
    let last_rel = (((base_row + rows) * row_bytes - 1) / 4) as u32;
    for rel_word in [
        first_rel,
        first_rel + 1,
        (first_rel + last_rel) / 2,
        last_rel,
    ] {
        let shader_local = wrapped.wrapping_add(rel_word);
        let expected_local = (weight_off + rel_word) - window.window_base_words();
        assert_eq!(
            shader_local, expected_local,
            "wrapped rebase must yield the window-local word for rel_word {}",
            rel_word
        );

        let (slot, offset) = window
            .absolute_to_local(window.window_base_words() + shader_local)
            .unwrap_or_else(|e| panic!("rel_word {} outside window: {}", rel_word, e));
        assert!(slot < window.slot_count, "rel_word {} unbound", rel_word);
        assert_eq!(
            window.local_to_absolute(slot, offset),
            weight_off + rel_word,
            "rel_word {} must round-trip to its absolute word",
            rel_word
        );
    }

    contract_test(
        "lm_head_tile_weight_off_rebase_wraps_correctly",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn lm_head_tile_windows_cover_every_vocab_row() {
    clear_invariant_log();

    // Sweep all TDR tiles across the vocab: every tile must be bindable, and
    // the tiles together must cover every row exactly once with no gap. A gap
    // means some vocab rows read from an unbound slot.
    let dim = 3_072u64;
    let vocab = 128_256u64;
    let row_bytes = head_row_bytes(dim, 256, 210);
    let weight_byte = 1_600_000_000u64;
    let weight_off = (weight_byte / 4) as u32;

    let tile_size = 64u64; // @workgroup_size in sh_head_blob.wgsl
    let total_wgs = vocab.div_ceil(tile_size);
    let max_safe_wgs = 256u64; // representative TDR budget

    let mut wgs_dispatched = 0u64;
    let mut covered_rows = 0u64;

    while wgs_dispatched < total_wgs {
        let this_tile = (total_wgs - wgs_dispatched).min(max_safe_wgs);
        let base_row = wgs_dispatched * tile_size;
        let rows = (this_tile * tile_size).min(vocab - base_row);

        let start_word = ((weight_byte + base_row * row_bytes) / 4) as u32;
        let end_word = ((weight_byte + (base_row + rows) * row_bytes - 1) / 4) as u32;

        let window = BlobWindow::for_range(start_word, end_word, CHUNK_WORDS_128_MIB, 16)
            .unwrap_or_else(|e| panic!("tile at base_row {} unbindable: {}", base_row, e));

        // First and last row of the tile must both resolve through the window.
        let wrapped = weight_off.wrapping_sub(window.window_base_words());
        for rel_byte in [base_row * row_bytes, (base_row + rows) * row_bytes - 1] {
            let shader_local = wrapped.wrapping_add((rel_byte / 4) as u32);
            let abs = window.window_base_words() + shader_local;
            assert!(
                window.contains(abs),
                "tile at base_row {} does not cover byte {}",
                base_row,
                rel_byte
            );
        }

        covered_rows += rows;
        wgs_dispatched += this_tile;
    }

    assert_eq!(
        covered_rows, vocab,
        "tiles must cover every vocab row exactly once"
    );

    contract_test("lm_head_tile_windows_cover_every_vocab_row", &[]);
}

// ---------------------------------------------------------------------------
// Bead `airframe-kqa` — dequant_any window algebra + the shader offset contract.
//
// sh_dequant_any.wgsl resolves an element as
//     word = off_pack / 2 + blob_base_words
// where `off_pack` is a PACKED (byte/2) offset. So a caller has exactly two
// consistent encodings, and they must not be mixed:
//   A) blob_base_words = byte/4, off_pack = 0     (used by every working path)
//   B) blob_base_words = 0,      off_pack = byte/2
// Passing a WORD index as off_pack is encoding (A)'s value into (B)'s slot and
// silently halves the offset. These tests pin the contract.
//
// `count` is also an ELEMENT count, so a dequant window must be sized from the
// quant type's block geometry, never treated as one word per element.
// ---------------------------------------------------------------------------

/// The shader's own address arithmetic: `gow(off_pack) = off_pack/2 + base`.
/// Returns the absolute byte the dispatch reads first.
fn dequant_shader_first_byte(blob_base_words: u32, off_pack: u32) -> u64 {
    (off_pack as u64 / 2) * 4 + (blob_base_words as u64) * 4
}

#[test]
fn dequant_any_offset_encoding_contract() {
    clear_invariant_log();

    // A tensor row at an arbitrary 4-byte-aligned offset.
    let offset_bytes = 1_234_567_890u64 & !3u64;

    // Encoding (A): everything in blob_base_words, off_pack = 0.
    let enc_a = dequant_shader_first_byte((offset_bytes / 4) as u32, 0);
    assert_eq!(
        enc_a, offset_bytes,
        "encoding A must address the tensor start exactly"
    );

    // Encoding (B): everything in the packed offset.
    let enc_b = dequant_shader_first_byte(0, (offset_bytes / 2) as u32);
    assert_eq!(
        enc_b, offset_bytes,
        "encoding B must address the tensor start exactly"
    );

    // The regression: passing a WORD index (byte/4) as off_pack. gow halves it
    // again, so the dispatch reads from HALF the intended offset.
    let mixed = dequant_shader_first_byte(0, (offset_bytes / 4) as u32);
    assert_eq!(
        mixed,
        offset_bytes / 2,
        "a word index in the packed-offset slot must be shown to halve the address"
    );
    assert_ne!(
        mixed, offset_bytes,
        "mixing encodings must not accidentally land on the right byte"
    );

    contract_test("dequant_any_offset_encoding_contract", &[]);
}

#[test]
fn dequant_any_window_rebase_addresses_tensor_start() {
    clear_invariant_log();

    // Embedding row for Llama-3.2-3B living past resident chunk 7 — the case
    // the window abstraction exists for. After rebasing, the shader's own
    // arithmetic must land on the tensor start, window-locally.
    let offset_bytes = 1_500_000_000u64 & !3u64;
    let offset_words = (offset_bytes / 4) as u32;
    let count = 3_072u32; // n_embd
    let q6k_bytes = head_row_bytes(count as u64, 256, 210);

    let last_word = offset_words as u64 + (q6k_bytes - 1) / 4;
    let window = BlobWindow::for_range(offset_words, last_word as u32, CHUNK_WORDS_128_MIB, 16)
        .expect("embedding row must be bindable");

    assert!(
        window.start_chunk > 7,
        "fixture must exercise a row past the 8 identity slots"
    );

    // Host-side rebase, exactly as run_dequant_any_hot does it.
    let local_base = offset_words - window.window_base_words();

    // Shader arithmetic on the rebased params yields the window-local byte…
    let shader_byte = dequant_shader_first_byte(local_base, 0);
    // …which maps back to the true absolute byte through the window base.
    let absolute = (window.window_base_words() as u64) * 4 + shader_byte;
    assert_eq!(
        absolute, offset_bytes,
        "rebased dispatch must address the true tensor start"
    );

    // Both ends of the row resolve to bound slots.
    for word in [offset_words, last_word as u32] {
        let (slot, offset) = window
            .absolute_to_local(word)
            .unwrap_or_else(|e| panic!("word {} outside window: {}", word, e));
        assert!(slot < window.slot_count, "word {} unbound", word);
        assert_eq!(window.local_to_absolute(slot, offset), word);
    }

    contract_test(
        "dequant_any_window_rebase_addresses_tensor_start",
        &["Word index must be within buffer bounds"],
    );
}

#[test]
fn dequant_window_span_uses_block_geometry_not_element_count() {
    clear_invariant_log();

    // `count` counts ELEMENTS. Treating it as words (4 B/elem) overstates a
    // Q4_K span ~7x and a Q6_K span ~4.9x, which can push an otherwise
    // bindable window past the resident chunk count.
    let count = 3_072u64;

    let q4k_bytes = head_row_bytes(count, 256, 144);
    let q6k_bytes = head_row_bytes(count, 256, 210);
    let naive_bytes = count * 4;

    assert_eq!(q4k_bytes, 1_728);
    assert_eq!(q6k_bytes, 2_520);
    assert!(
        naive_bytes > q4k_bytes * 7,
        "element-as-word must be shown to grossly overstate a Q4_K span"
    );

    // A partial trailing block still reads a whole block from the file.
    let partial = 300u64; // 1 full Q4_K block + 44 elements
    let blocks = partial.div_ceil(256);
    assert_eq!(blocks, 2, "partial block must round up");
    assert_eq!(blocks * 144, 288, "span must cover both whole blocks");

    contract_test(
        "dequant_window_span_uses_block_geometry_not_element_count",
        &[],
    );
}

#[test]
fn dequant_window_chunk_zero_is_identity() {
    clear_invariant_log();

    // Small models keep embeddings in chunk 0; the rebase must be a no-op so
    // the single-chunk path is bit-identical to the pre-window behaviour.
    let offset_words = 8_192u32;
    let count = 2_048u64;
    let span = head_row_bytes(count, 256, 144);

    let window = BlobWindow::for_range(
        offset_words,
        (offset_words as u64 + (span - 1) / 4) as u32,
        CHUNK_WORDS_128_MIB,
        16,
    )
    .expect("valid window");

    assert_eq!(window.start_chunk, 0);
    assert_eq!(window.window_base_words(), 0);

    let local_base = offset_words - window.window_base_words();
    assert_eq!(local_base, offset_words, "chunk-0 rebase must be a no-op");
    assert_eq!(
        dequant_shader_first_byte(local_base, 0),
        (offset_words as u64) * 4,
        "chunk-0 dispatch must address the tensor start directly"
    );

    contract_test("dequant_window_chunk_zero_is_identity", &[]);
}
