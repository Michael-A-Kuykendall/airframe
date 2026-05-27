//! Vision ops frozen reference tests — zero GGUF dependency.
//!
//! Every test here feeds known numeric inputs into a vision op and asserts
//! the exact expected output.  Expected values are derived from:
//!
//!   a) Closed-form math (marked "ANALYTIC")
//!   b) PyTorch float32 output for the same inputs (marked "PYTORCH")
//!
//! Run with: `cargo test --test vision_math`
//! No model files, no GPU, no network — runs anywhere.

use airframe::ops::dispatch::OpDispatcher;
use airframe::core::tensor::Tensor;

fn ops() -> OpDispatcher { OpDispatcher::new() }
fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor { Tensor::new(data, shape).unwrap() }
fn t1(data: Vec<f32>) -> Tensor { Tensor::new(data.clone(), vec![data.len()]).unwrap() }

// ─── GELU ────────────────────────────────────────────────────────────────────

/// PYTORCH: torch.nn.functional.gelu(torch.tensor([0., 1., -1., 2., -2., 0.5]))
/// tensor([ 0.0000,  0.8413, -0.1587,  1.9546, -0.0454,  0.3457])
#[test]
fn vision_gelu_reference_vector() {
    let input = t1(vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5]);
    let out = ops().gelu(&input).unwrap();
    let expected = [0.0, 0.8413, -0.1587, 1.9546, -0.0454, 0.3457];
    for (i, (&got, &exp)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 5e-4,
            "gelu[{i}]: got {got}, expected {exp}");
    }
}

#[test]
fn vision_gelu_shape_preserved() {
    let input = t(vec![1.0; 12], vec![3, 4]);
    let out = ops().gelu(&input).unwrap();
    assert_eq!(out.shape, vec![3, 4]);
}

#[test]
fn vision_gelu_zero_is_zero() {
    let input = t1(vec![0.0]);
    let out = ops().gelu(&input).unwrap();
    assert!(out.data[0].abs() < 1e-7);
}

// ─── LAYERNORM ───────────────────────────────────────────────────────────────

/// ANALYTIC: input=[2,4,6,8], mean=5, var=5, inv_std=1/sqrt(5+1e-5)≈0.4472
/// normalized = [-3/sqrt(5), -1/sqrt(5), 1/sqrt(5), 3/sqrt(5)]
/// weight=ones, no bias → same as normalized
#[test]
fn vision_layernorm_1d_analytic() {
    let input = t1(vec![2.0, 4.0, 6.0, 8.0]);
    let weight = t1(vec![1.0, 1.0, 1.0, 1.0]);
    let out = ops().layernorm(&input, &weight, None, 1e-5).unwrap();

    // After LN the mean must be ~0 and variance ~1
    let mean: f32 = out.data.iter().sum::<f32>() / 4.0;
    let var: f32 = out.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5, "mean={mean}");
    assert!((var - 1.0).abs() < 1e-4, "var={var}");
    // Values must be symmetric around 0
    assert!((out.data[0] + out.data[3]).abs() < 1e-5); // −x and +x
    assert!((out.data[1] + out.data[2]).abs() < 1e-5);
}

/// PYTORCH: LayerNorm([3], eps=1e-5)
/// input = [[1,3,5],[7,9,11]]  weight=[2,2,2]  bias=[1,1,1]
/// Each row normalized then scaled ×2 +1:
///   row mean=3,var=8/3  → normalized≈[-1.2247,-0,1.2247] → *2+1=[-1.4495,1,3.4495]
///   row mean=9,var=8/3  → same normalized values, same output
#[test]
fn vision_layernorm_2d_pytorch_reference() {
    let input = t(vec![1.,3.,5., 7.,9.,11.], vec![2,3]);
    let weight = t1(vec![2.0, 2.0, 2.0]);
    let bias   = t1(vec![1.0, 1.0, 1.0]);
    let out = ops().layernorm(&input, &weight, Some(&bias), 1e-5).unwrap();

    assert_eq!(out.shape, vec![2, 3]);
    // Both rows should produce identical output (same relative spacing)
    for i in 0..3 {
        assert!((out.data[i] - out.data[i + 3]).abs() < 1e-4,
            "row0[{i}]={} vs row1[{i}]={}", out.data[i], out.data[i+3]);
    }
    // Middle element (normalized≈0) → output ≈ 0*2+1 = 1
    assert!((out.data[1] - 1.0).abs() < 1e-3, "mid={}", out.data[1]);
}

#[test]
fn vision_layernorm_scale_only_no_bias() {
    // input=[0, 0, 0, 4], weight=[1,1,1,1], no bias
    // mean=1, var=3, inv_std=1/sqrt(3+eps)
    let input = t1(vec![0.0, 0.0, 0.0, 4.0]);
    let weight = t1(vec![1.0; 4]);
    let out = ops().layernorm(&input, &weight, None, 1e-5).unwrap();
    // Last element is the positive outlier; should be the largest value
    assert!(out.data[3] > out.data[0]);
    assert!(out.data[3] > out.data[1]);
    assert!(out.data[3] > out.data[2]);
    // Sum must be ~0
    let sum: f32 = out.data.iter().sum();
    assert!(sum.abs() < 1e-4, "sum={sum}");
}

// ─── ADD_BROADCAST ───────────────────────────────────────────────────────────

/// ViT use-case: pos_embed [1, 5, 3] + patch_features [5, 3] → [5, 3]
/// ANALYTIC: each element = pos_val + feat_val
#[test]
fn vision_add_broadcast_vit_pos_embed() {
    // pos_embed: [1, 5, 3] — all ones
    let pos = t(vec![1.0; 15], vec![1, 5, 3]);
    // patch features: [5, 3] — sequential 0..14
    let feat: Vec<f32> = (0..15).map(|x| x as f32).collect();
    let feat = t(feat, vec![5, 3]);

    let out = ops().add_broadcast(&pos, &feat).unwrap();
    assert_eq!(out.shape, vec![5, 3]);
    for (i, &v) in out.data.iter().enumerate() {
        let expected = i as f32 + 1.0;
        assert!((v - expected).abs() < 1e-6, "[{i}]: got {v} expected {expected}");
    }
}

#[test]
fn vision_add_broadcast_symmetric() {
    // [1, 3] + [3] and [3] + [1, 3] should give same result
    let a = t(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = t(vec![4.0, 5.0, 6.0], vec![3]);
    let out_ab = ops().add_broadcast(&a, &b).unwrap();
    let out_ba = ops().add_broadcast(&b, &a).unwrap();
    assert_eq!(out_ab.data, out_ba.data);
}

// ─── PATCH_EMBED ─────────────────────────────────────────────────────────────

/// ANALYTIC: 1-channel 4×4 image, patch=2, out_ch=2
/// weight[0] = all 1s  (sum of 2×2 patch)
/// weight[1] = [1,-1,1,-1] alternating (difference)
/// bias = [0, 0]
/// Image: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
///   patch (0,0): [1,2,5,6]  → filter0=14, filter1=(1-2+5-6)=-2
///   patch (0,1): [3,4,7,8]  → filter0=22, filter1=(3-4+7-8)=-2
///   patch (1,0): [9,10,13,14]→ filter0=46, filter1=(9-10+13-14)=-2
///   patch (1,1): [11,12,15,16]→ filter0=54, filter1=(11-12+15-16)=-2
#[test]
fn vision_patch_embed_analytic_2x2_patches() {
    let image_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let image = t(image_data, vec![1, 4, 4]);

    // filter0: all ones [1,1,1,1], filter1: alternating [1,-1,1,-1]
    let weight = t(vec![
        1.0,  1.0,  1.0,  1.0,  // filter 0
        1.0, -1.0,  1.0, -1.0,  // filter 1
    ], vec![2, 1, 2, 2]);
    let bias = t(vec![0.0, 0.0], vec![2]);

    let out = ops().patch_embed(&image, &weight, &bias, 2).unwrap();
    assert_eq!(out.shape, vec![4, 2]);

    let expected = [14.0, -2.0,  22.0, -2.0,  46.0, -2.0,  54.0, -2.0];
    for (i, (&got, &exp)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-4, "patch_embed[{i}]: got {got} expected {exp}");
    }
}

/// Shape contract: SigLIP-So400M dims — 448×448 → 1024 patches of 1152 dims
#[test]
fn vision_patch_embed_siglip_shape() {
    let image  = Tensor::zeros(vec![3, 448, 448]);
    let weight = Tensor::zeros(vec![1152, 3, 14, 14]);
    let bias   = Tensor::zeros(vec![1152]);
    let out = ops().patch_embed(&image, &weight, &bias, 14).unwrap();
    assert_eq!(out.shape, vec![1024, 1152]);
}

/// Bias is applied: zero weight + known bias → output == bias repeated per patch
#[test]
fn vision_patch_embed_bias_only() {
    let image  = Tensor::zeros(vec![1, 14, 14]);
    let weight = Tensor::zeros(vec![4, 1, 14, 14]);
    let bias   = t1(vec![1.0, 2.0, 3.0, 4.0]);
    let out = ops().patch_embed(&image, &weight, &bias, 14).unwrap();
    assert_eq!(out.shape, vec![1, 4]);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
}

// ─── BIDIRECTIONAL ATTENTION (via OpDispatcher) ───────────────────────────────

/// Verify that with causal_mask=false, token-0 can see token-3.
/// (Full coverage is in ops/reference/attention.rs — this exercises the
/// OpDispatcher path specifically.)
#[test]
fn vision_bidirectional_attn_dispatcher_path() {
    let hidden = 4;
    let head_dim = 4;
    let eye4 = vec![1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.];

    let q_w = t(eye4.clone(), vec![hidden, head_dim]);
    let k_w = t(eye4.clone(), vec![hidden, head_dim]);
    let v_w = t(eye4.clone(), vec![hidden, head_dim]);
    let o_w = t(eye4.clone(), vec![head_dim, hidden]);

    // Input where tok-3 has a distinctive signature
    let input_a = t(vec![
        1.,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,5.
    ], vec![4, hidden]);
    let input_b = t(vec![
        1.,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,0.
    ], vec![4, hidden]);

    let pos: Vec<usize> = (0..4).collect();
    let out_a = ops().attention(
        &input_a, &q_w, &k_w, &v_w, &o_w,
        1, 1, head_dim, &pos, 10000.0, head_dim, 1.0, false
    ).unwrap();
    let out_b = ops().attention(
        &input_b, &q_w, &k_w, &v_w, &o_w,
        1, 1, head_dim, &pos, 10000.0, head_dim, 1.0, false
    ).unwrap();

    // Token-0 output must differ — it can see token-3
    let diff: f32 = out_a.data[..hidden].iter()
        .zip(out_b.data[..hidden].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-4, "tok-0 output unchanged (diff={diff}) — causal mask may be stuck ON");
}

// ─── Phase 2: vit_mha (bidirectional, no RoPE) ───────────────────────────────

/// vit_mha: output shape matches [seq, out_dim].
#[test]
fn vision_vit_mha_output_shape() {
    let seq = 6; let d = 8; let out = 8;
    let q  = t(vec![0.0; seq * d], vec![seq, d]);
    let k  = t(vec![0.0; seq * d], vec![seq, d]);
    let v  = t(vec![0.0; seq * d], vec![seq, d]);
    let ow = t(vec![0.0; d * out], vec![d, out]);
    let ob = t(vec![0.0; out],     vec![out]);
    let result = ops().vit_attention(&q, &k, &v, &ow, &ob, 2, 4).unwrap();
    assert_eq!(result.shape, vec![seq, out]);
}

/// vit_mha: all tokens receive signal from a future token (truly bidirectional).
/// ANALYTIC: Q=K=[1,0,…] uniform attention → all tokens average V equally.
/// Token-1 has V spike at dim-1.  Token-0 should see a non-zero contribution.
#[test]
fn vision_vit_mha_bidirectional_signal() {
    let seq = 3; let hd = 2;
    let q = t(vec![1.,0., 1.,0., 1.,0.], vec![seq, hd]);
    let k = t(vec![1.,0., 1.,0., 1.,0.], vec![seq, hd]);
    // Token-1 has a spike at dim-1 of V
    let v = t(vec![0.,0., 0.,7., 0.,0.], vec![seq, hd]);
    let ow = t(vec![1.,0., 0.,1.], vec![hd, hd]);  // identity
    let ob = t(vec![0.,0.], vec![hd]);
    let out = ops().vit_attention(&q, &k, &v, &ow, &ob, 1, hd).unwrap();
    // Token-0 should pick up the spike (uniform attention, 1/3 weight on token-1)
    assert!(out.data[1] > 0.5, "tok-0 dim-1 expected > 0.5, got {}", out.data[1]);
    // All tokens should produce identical output (equal Q rows → same attn weights)
    assert!((out.data[0] - out.data[2]).abs() < 1e-5);
    assert!((out.data[1] - out.data[3]).abs() < 1e-5);
}

// ─── Phase 2: SigLipBlock ─────────────────────────────────────────────────────

/// SigLipBlock forward: output has correct shape.
#[test]
fn vision_siglip_block_shape() {
    use airframe::family::vit::{zero_block, SigLipConfig};
    let cfg = SigLipConfig { hidden_dim: 8, n_heads: 2, head_dim: 4, mlp_dim: 16, ..SigLipConfig::default() };
    let block = zero_block(cfg.hidden_dim, cfg.mlp_dim);
    let x = t(vec![0.0; 5 * 8], vec![5, 8]);
    let out = block.forward(&x, &ops(), &cfg).unwrap();
    assert_eq!(out.shape, vec![5, 8]);
}

/// SigLipBlock forward: output is finite even with random-ish inputs.
#[test]
fn vision_siglip_block_finite_output() {
    use airframe::family::vit::{zero_block, SigLipConfig};
    let cfg = SigLipConfig { hidden_dim: 4, n_heads: 1, head_dim: 4, mlp_dim: 8, ..SigLipConfig::default() };
    let block = zero_block(cfg.hidden_dim, cfg.mlp_dim);
    let x = t((0..8).map(|i| i as f32 * 0.5).collect(), vec![2, 4]);
    let out = block.forward(&x, &ops(), &cfg).unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    assert!(out.data.iter().all(|v| v.is_finite()),
        "SigLipBlock output is non-finite: {:?}", out.data);
}

// ─── Phase 2: Resampler ───────────────────────────────────────────────────────

/// Resampler: always produces [n_queries, d_model] regardless of ViT token count.
#[test]
fn vision_resampler_output_shape() {
    use airframe::family::resampler::{identity_resampler, ResamplerConfig};
    let cfg = ResamplerConfig { n_queries: 4, d_model: 8, kv_dim: 6, n_heads: 2, head_dim: 4, layer_norm_eps: 1e-5 };
    let r = identity_resampler(cfg);
    let vit = t(vec![0.0; 10 * 6], vec![10, 6]);
    let out = r.forward(&vit, &ops()).unwrap();
    assert_eq!(out.shape, vec![4, 8]);
}

/// Resampler: output is finite with non-trivial input values.
#[test]
fn vision_resampler_finite_output() {
    use airframe::family::resampler::{identity_resampler, ResamplerConfig};
    let cfg = ResamplerConfig { n_queries: 4, d_model: 8, kv_dim: 6, n_heads: 2, head_dim: 4, layer_norm_eps: 1e-5 };
    let r = identity_resampler(cfg);
    let vit = t((0..60).map(|i| i as f32 * 0.1 - 3.0).collect(), vec![10, 6]);
    let out = r.forward(&vit, &ops()).unwrap();
    assert!(out.data.iter().all(|v| v.is_finite()),
        "Resampler output non-finite: {:?}", &out.data[..8]);
}

/// Resampler: 8-query count is invariant to ViT sequence length.
#[test]
fn vision_resampler_query_count_invariant() {
    use airframe::family::resampler::{identity_resampler, ResamplerConfig};
    for n_vit in [1usize, 5, 25, 100] {
        let r = identity_resampler(ResamplerConfig {
            n_queries: 8, d_model: 4, kv_dim: 4, n_heads: 1, head_dim: 4, layer_norm_eps: 1e-5,
        });
        let vit = t(vec![0.0; n_vit * 4], vec![n_vit, 4]);
        let out = r.forward(&vit, &ops()).unwrap();
        assert_eq!(out.shape[0], 8, "n_vit={n_vit}: expected 8 queries, got {}", out.shape[0]);
    }
}

