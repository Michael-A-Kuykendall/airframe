//! f41.3 verification: proportional RoPE factor direction.
//!
//! llama.cpp `ggml_rope_cache_init` (pinned: 49f35421, ggml-cpu/ops.cpp:5707)
//! applies per-pair freq factors as `theta / ff` (DIVIDE). Airframe's
//! `compute_rope_table_cfg` must match that convention — gemma-4's
//! `rope_freqs.weight` encodes divide factors.
//!
//! Airframe uses theta_i = base^(-2i/dim); llama.cpp uses a running product
//! starting at base. Both are valid theta conventions. What matters for
//! gemma-4 is the FACTOR direction: this test isolates it by using airframe's
//! theta convention and varying only whether the factor divides or multiplies.
//!
//! Run: cargo test --package airframe --test rope_factor_verify -- --nocapture

use airframe::backend::bindless::preflight::PreflightResources;
use airframe::core::spec::{GgufFileType, ModelArch, ModelSpec, NormKind};

/// Reference table using airframe's theta convention with factor applied as
/// DIVIDE (matching llama.cpp `ggml_rope_cache_init`, ops.cpp:5715 `theta/ff`).
/// Layout matches airframe's:
///   table[d * n_pairs * 2 + p * 2 + 0] = cos(d * effective_theta_p)
///   table[d * n_pairs * 2 + p * 2 + 1] = sin(d * effective_theta_p)
fn reference_divide(base: f32, dim: usize, n_ctx: usize, factors: Option<&[f32]>) -> Vec<f32> {
    let n_pairs = dim / 2;
    let effective_thetas: Vec<f32> = (0..n_pairs)
        .map(|i| {
            let theta = 1.0_f32 / base.powf((2.0 * i as f32) / dim as f32);
            match factors {
                Some(f) if i < f.len() => theta / f[i], // DIVIDE — correct
                _ => theta,
            }
        })
        .collect();
    let mut table = Vec::with_capacity(n_ctx * n_pairs * 2);
    for d in 0..n_ctx {
        for &t in &effective_thetas {
            let angle = d as f32 * t;
            table.push(angle.cos());
            table.push(angle.sin());
        }
    }
    table
}

/// Same theta convention, factor applied as MULTIPLY — the pre-fix convention.
fn reference_multiply(base: f32, dim: usize, n_ctx: usize, factors: Option<&[f32]>) -> Vec<f32> {
    let n_pairs = dim / 2;
    let effective_thetas: Vec<f32> = (0..n_pairs)
        .map(|i| {
            let theta = 1.0_f32 / base.powf((2.0 * i as f32) / dim as f32);
            match factors {
                Some(f) if i < f.len() => theta * f[i], // MULTIPLY — buggy
                _ => theta,
            }
        })
        .collect();
    let mut table = Vec::with_capacity(n_ctx * n_pairs * 2);
    for d in 0..n_ctx {
        for &t in &effective_thetas {
            let angle = d as f32 * t;
            table.push(angle.cos());
            table.push(angle.sin());
        }
    }
    table
}

fn test_spec(n_ctx: usize) -> ModelSpec {
    ModelSpec {
        n_vocab: 32,
        n_embd: 8,
        n_layer: 1,
        n_head: 2,
        n_head_kv: 1,
        ff_dim: 16,
        rms_eps: 1e-5,
        norm_kind: NormKind::Rms,
        rope_base: 10000.0,
        rope_scale: 1.0,
        rope_dim: 4,
        yarn_alpha: 1.0,
        yarn_beta: 32.0,
        n_ctx,
        head_dim: 4,
        gqa_ratio: 2,
        kv_dim: 4,
        max_head_dim: 0,
        arch: ModelArch::Gemma,
        file_type: GgufFileType::Q4_0,
        model_name: "rope-verify".to_string(),
        chat_template: None,
        temp_buffer_size: 64,
        kv_cache_size_per_layer: 64,
        attn_logit_softcap: 0.0,
        final_logit_softcap: 0.0,
        has_qk_norm: false,
        post_norm_enabled: false,
        q_weight_k: 0,
        k_weight_k: 0,
        dense_latent_layout: false,
        latent_dim: 0,
        per_layer_token_embd_offset: 0,
        per_layer_token_embd_quant: 0,
        per_layer_model_proj_offset: 0,
        per_layer_proj_norm_offset: 0,
        v_plain_rms_norm: false,
        out_scale_enabled: false,
        scale_embeddings_by_sqrt_dim: false,
        ple_inp_gate_offset: 0,
        ple_inp_gate_quant: 0,
        ple_proj_offset: 0,
        ple_proj_quant: 0,
        ple_layer_output_scale_offset: 0,
        ple_rope_freqs_offset: 0,
        ple_attn_post_norm_offset: 0,
        ple_ffn_post_norm_offset: 0,
        ple_post_norm_offset: 0,
        ple_latent_dim: 0,
        ple_enabled: false,
    }
}

/// Airframe must produce the DIVIDE table, never the buggy MULTIPLY table.
#[test]
fn rope_factor_direction_matches_llama_divide() {
    let dim = 8;
    let n_ctx = 4;
    let base = 1e6; // gemma-4 uses freq_base ≈ 1e6
                    // Distinct, non-unit factors so multiply vs divide is unambiguous.
    let factors: Vec<f32> = vec![1.0, 2.0, 0.5, 4.0];

    let spec = test_spec(n_ctx);
    let airframe_table =
        PreflightResources::compute_rope_table_cfg(&spec, 1.0, dim, base, Some(&factors));

    let divide_ref = reference_divide(base, dim, n_ctx, Some(&factors));
    let multiply_ref = reference_multiply(base, dim, n_ctx, Some(&factors));

    // Airframe must match the DIVIDE reference within float tolerance.
    assert_eq!(airframe_table.len(), divide_ref.len());
    for i in 0..airframe_table.len() {
        let diff = (airframe_table[i] - divide_ref[i]).abs();
        assert!(
            diff < 1e-5,
            "index {}: airframe={:.8} divide_ref={:.8} diff={:.2e}",
            i,
            airframe_table[i],
            divide_ref[i],
            diff
        );
    }

    // And must differ meaningfully from the buggy MULTIPLY table
    // (sanity check that the test actually discriminates the direction).
    let max_multiply_diff: f32 = airframe_table
        .iter()
        .zip(multiply_ref.iter())
        .map(|(a, m)| (a - m).abs())
        .fold(0.0, f32::max);
    assert!(
        max_multiply_diff > 0.1,
        "airframe table should NOT match multiply convention, but max diff is only {max_multiply_diff}"
    );
}

/// With all factors = 1.0, divide and multiply are identical — regression
/// guard: a passing implementation with unit factors does not prove direction,
/// but a FAILING one proves breakage.
#[test]
fn rope_unit_factors_table_is_consistent() {
    let dim = 8;
    let n_ctx = 4;
    let base = 1e6;
    let factors = vec![1.0f32; dim / 2];

    let spec = test_spec(n_ctx);
    let airframe_table =
        PreflightResources::compute_rope_table_cfg(&spec, 1.0, dim, base, Some(&factors));
    let reference_table = reference_divide(base, dim, n_ctx, Some(&factors));

    assert_eq!(airframe_table.len(), reference_table.len());
    for i in 0..airframe_table.len() {
        let diff = (airframe_table[i] - reference_table[i]).abs();
        assert!(
            diff < 1e-5,
            "index {}: airframe={:.8} ref={:.8} diff={:.2e}",
            i,
            airframe_table[i],
            reference_table[i],
            diff
        );
    }
}
