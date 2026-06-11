/// Property test: temp_buffer_size must be large enough for the full GPU temp layout.
///
/// Root cause this prevents:
///   TinyLlama (and any model with GQA) requires temp buffer of:
///     n_embd + q_len + kv_len*2 + ff_dim*2
///   The old formula only allocated:
///     max(n_embd*4, ff_dim*2 + n_embd)  → 13312 for TinyLlama
///   The correct minimum is 15872, so the GPU wrote past buffer end → all-NaN.
///
/// This test covers ALL models in the known spec library.

use airframe::core::spec::ModelSpec;

fn assert_temp_buffer_sufficient(spec: &ModelSpec, label: &str) {
    let q_len = spec.n_head * spec.head_dim;
    let kv_len = spec.n_head_kv * spec.head_dim;
    let required = spec.n_embd + q_len + kv_len * 2 + spec.ff_dim * 2;
    assert!(
        spec.temp_buffer_size >= required,
        "TEMP BUFFER UNDERALLOCATED for {}:\n  spec.temp_buffer_size = {}\n  required              = {}\n  shortage              = {}\n  formula: n_embd({}) + q_len({}) + kv_len*2({}) + ff_dim*2({})",
        label,
        spec.temp_buffer_size,
        required,
        required.saturating_sub(spec.temp_buffer_size),
        spec.n_embd, q_len, kv_len * 2, spec.ff_dim * 2
    );
}

#[test]
fn tinyllama_temp_buffer_sufficient() {
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    assert_temp_buffer_sufficient(&spec, "TinyLlama-1.1B-Chat-v1.0");

    // Exact expected values after fix
    assert_eq!(spec.temp_buffer_size, 16384,
        "TinyLlama temp_buffer_size should be 16384 after fix (was 13312)");
}

#[test]
fn all_known_specs_temp_buffer_sufficient() {
    // Build all known specs via metadata path (spec-from-gguf uses compute_derived too)
    let specs: Vec<(&str, ModelSpec)> = vec![
        ("TinyLlama-1.1B", ModelSpec::tinylama_1_1b_chat_v1_0()),
    ];

    for (label, spec) in &specs {
        assert_temp_buffer_sufficient(spec, label);
    }
}

#[test]
fn temp_buffer_formula_matches_test_hardcoded_value() {
    // The test_layer_dump.rs and test_parity.rs use hardcoded temp_stride: 16384.
    // This test verifies the spec formula produces that same value for TinyLlama.
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    assert_eq!(
        spec.temp_buffer_size, 16384,
        "spec.temp_buffer_size must match the hardcoded 16384 used in test_layer_dump.rs and test_parity.rs"
    );
}

#[test]
fn temp_buffer_formula_is_monotone_in_model_size() {
    // Larger models (more heads, bigger FFN) must always get larger temp buffers.
    // Construct two synthetic specs and verify the formula scales correctly.
    use airframe::core::spec::{GgufFileType, ModelArch};

    let small = ModelSpec {
        n_vocab: 32000, n_embd: 2048, n_layer: 22, n_head: 32, n_head_kv: 4,
        ff_dim: 5632, rms_eps: 1e-5, rope_base: 10000.0, rope_scale: 1.0,
        rope_dim: 64, yarn_alpha: 1.0, yarn_beta: 32.0, n_ctx: 2048,
        attn_logit_softcap: 0.0, final_logit_softcap: 0.0, has_qk_norm: false,
        head_dim: 0, gqa_ratio: 0, kv_dim: 0,
        arch: ModelArch::Llama, file_type: GgufFileType::Q4_0,
        model_name: "small_test".to_string(), temp_buffer_size: 0, kv_cache_size_per_layer: 0,
    }.compute_derived();

    let large = ModelSpec {
        n_vocab: 32000, n_embd: 4096, n_layer: 32, n_head: 32, n_head_kv: 8,
        ff_dim: 11008, rms_eps: 1e-5, rope_base: 10000.0, rope_scale: 1.0,
        rope_dim: 128, yarn_alpha: 1.0, yarn_beta: 32.0, n_ctx: 4096,
        attn_logit_softcap: 0.0, final_logit_softcap: 0.0, has_qk_norm: false,
        head_dim: 0, gqa_ratio: 0, kv_dim: 0,
        arch: ModelArch::Llama, file_type: GgufFileType::Q4_0,
        model_name: "large_test".to_string(), temp_buffer_size: 0, kv_cache_size_per_layer: 0,
    }.compute_derived();

    assert!(
        large.temp_buffer_size > small.temp_buffer_size,
        "Larger model must have larger temp_buffer_size: large={} small={}",
        large.temp_buffer_size, small.temp_buffer_size
    );

    assert_temp_buffer_sufficient(&small, "small_test");
    assert_temp_buffer_sufficient(&large, "large_test");
}
