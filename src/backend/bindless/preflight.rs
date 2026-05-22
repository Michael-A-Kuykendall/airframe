use crate::core::spec::ModelSpec;
use wgpu::util::DeviceExt;

/// The "Pre-Flight" deck.
/// Contains resources that are "Fused" (Pre-computed/Extracted)
/// from the raw model to prepare for high-speed GPU execution.
pub struct PreflightResources {
    /// Pre-computed RoPE frequencies (Cos/Sin pairs), runtime buffer.
    /// Layout: [distance][pair] each entry = (cos, sin).
    /// Written via write_buffer before each inference to select native or extended table.
    pub rope_cache_buffer: wgpu::Buffer,

    /// Raw RoPE table data at native scale (scale=1.0, no frequency modification).
    /// Used for requests whose total sequence length fits within the training context.
    pub rope_data_native: Vec<f32>,

    /// Raw RoPE table data at the configured extended scale (may equal native if rope_scale=1.0).
    /// Used for requests that genuinely exceed the training context.
    pub rope_data_ext: Vec<f32>,

    /// Extracted and Aligned Norm Weights.
    /// Eliminates the need for unaligned reads in the shader.
    /// Layout: [Layer 0 AttnNorm] [Layer 0 FFN Norm] ... [Final Norm]
    pub norm_bank_buffer: wgpu::Buffer,
}

impl PreflightResources {
    pub fn new_from_ram(
        device: &wgpu::Device,
        raw_data: &[u8],
        metadata: &super::metadata::BindlessMetadata,
        spec: &ModelSpec,
    ) -> Self {
        println!("[Preflight] Spinning off Math Tables & Tensor Banks (From RAM)...");

        let rope_data_ext = Self::compute_rope_table(spec, spec.rope_scale);
        let rope_data_native = if spec.rope_scale < 1.0 {
            Self::compute_rope_table(spec, 1.0)
        } else {
            rope_data_ext.clone()
        };
        let rope_buffer = Self::upload_rope_table(device, &rope_data_ext, spec.rope_scale, spec);
        let norm_buffer = Self::build_norm_bank_from_ram(device, raw_data, metadata, spec);

        Self {
            rope_cache_buffer: rope_buffer,
            rope_data_native,
            rope_data_ext,
            norm_bank_buffer: norm_buffer,
        }
    }

    /// Compute a RoPE cos/sin lookup table with the given scale override.
    ///
    /// YaRN (Peng et al. 2023) applies a non-uniform frequency correction per
    /// RoPE dimension pair:
    ///
    ///   For pair i with base theta_i = 1 / base^(2i/D):
    ///     wavelength lambda_i = 2*pi / theta_i = 2*pi * base^(2i/D)
    ///     low  = L_train / yarn_alpha  (long-wavelength dims: no scaling needed)
    ///     high = L_train / yarn_beta   (short-wavelength dims: full linear scaling)
    ///
    ///   ramp_i = clamp((L_train / lambda_i - yarn_alpha) / (yarn_beta - yarn_alpha), 0.0, 1.0)
    ///   effective_theta_i = theta_i * (ramp_i / rope_scale + (1.0 - ramp_i))
    ///     = theta_i when ramp=0 (low freq, no scaling)
    ///     = theta_i / rope_scale when ramp=1 (high freq, full linear equiv)
    ///
    /// Layout: [distance][pair] each entry = (cos, sin).
    ///   table[d * n_pairs * 2 + p * 2 + 0] = cos(d * effective_theta_p)
    ///   table[d * n_pairs * 2 + p * 2 + 1] = sin(d * effective_theta_p)
    fn compute_rope_table(spec: &ModelSpec, scale: f32) -> Vec<f32> {
        let dim = spec.rope_dim;
        let base = spec.rope_base;
        let n_pairs = dim / 2;
        let max_dist = spec.n_ctx;

        let l_train = if scale < 1.0 { (max_dist as f32 * scale).round() as usize } else { max_dist };
        let alpha = spec.yarn_alpha;
        let beta = spec.yarn_beta;
        let use_yarn = scale < 1.0 && alpha > 0.0 && beta > alpha;

        // Compute per-dimension effective theta with optional YaRN correction.
        let effective_thetas: Vec<f32> = (0..n_pairs)
            .map(|i| {
                let theta = 1.0_f32 / base.powf((2.0 * i as f32) / dim as f32);
                if !use_yarn {
                    return theta * scale; // plain linear or native
                }
                let lambda = std::f32::consts::TAU / theta; // wavelength = 2*pi / theta
                // ramp: 1.0 at short wavelengths, 0.0 at long wavelengths.
                // Per YaRN: high-freq dims skip scaling; low-freq dims get full linear scaling.
                // Lerp: ramp drives theta toward 1.0 for high-freq, toward scale for low-freq.
                let ramp = ((l_train as f32 / lambda - alpha) / (beta - alpha)).clamp(0.0, 1.0);
                theta * ((1.0 - ramp) * scale + ramp)
            })
            .collect();

        let table_len = max_dist * n_pairs * 2;
        let mut table = Vec::with_capacity(table_len);
        for d in 0..max_dist {
            for p in 0..n_pairs {
                let angle = d as f32 * effective_thetas[p];
                table.push(angle.cos());
                table.push(angle.sin());
            }
        }
        table
    }

    fn upload_rope_table(device: &wgpu::Device, rope_floats: &[f32], scale: f32, spec: &ModelSpec) -> wgpu::Buffer {
        let n_pairs = spec.rope_dim / 2;
        let max_dist = spec.n_ctx;
        let table_len = max_dist * n_pairs * 2;
        let use_yarn = scale < 1.0 && spec.yarn_alpha > 0.0 && spec.yarn_beta > spec.yarn_alpha;
        let scaling_mode = if scale >= 1.0 { "native" } else if use_yarn { "YaRN" } else { "linear" };
        println!(
            "[Preflight] RoPE Lookup Table: {}×{} = {} entries ({:.1} KB, Base={}, Dim={}, Scale={}, Mode={})",
            max_dist, n_pairs, table_len,
            (table_len * 4) as f64 / 1024.0,
            spec.rope_base, spec.rope_dim, scale, scaling_mode
        );
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RoPE Cos/Sin Lookup Table"),
            contents: bytemuck::cast_slice(rope_floats),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn build_norm_bank_from_ram(
        device: &wgpu::Device,
        raw_data: &[u8],
        metadata: &super::metadata::BindlessMetadata,
        spec: &ModelSpec,
    ) -> wgpu::Buffer {
        // We need to extract:
        // 1. attn_norm — one per layer
        // 2. ffn_norm — one per layer
        // 3. output_norm — final layer

        let dim = spec.n_embd;
        let n_layers = spec.n_layer;

        // Block size = dim * 4 bytes
        let block_size = dim * 4;

        // Layout:
        // Layers: [AttnNorm, FfnNorm, PostAttnNorm, PostFfwNorm] interleaved (4 slots/layer)
        // End: [OutputNorm]
        let total_size = (n_layers * 4 + 1) * (block_size as usize);

        let mut bank = vec![0u8; total_size];

        println!(
            "[Preflight] Extracting Norms: {} layers, Total Size {:.2} MB",
            n_layers,
            total_size as f64 / 1024.0 / 1024.0
        );

        // Helper to copy
        let mut copy_tensor = |name: &str, dest_offset_bytes: usize| {
            if let Some(offset) = metadata.get_tensor_offset(name) {
                let start = offset as usize;
                let end = start + (block_size as usize);
                if end > raw_data.len() {
                    eprintln!("CRITICAL ERROR: Tensor {} out of bounds!", name);
                } else {
                    bank[dest_offset_bytes..dest_offset_bytes + (block_size as usize)]
                        .copy_from_slice(&raw_data[start..end]);
                }
            } else {
                eprintln!("WARNING: Missing Norm Tensor {}", name);
            }
        };

        for i in 0..n_layers {
            let attn_name = format!("blk.{}.attn_norm.weight", i);
            let ffn_name = format!("blk.{}.ffn_norm.weight", i);
            let post_attn_name = format!("blk.{}.post_attention_norm.weight", i);
            let post_ffw_name = format!("blk.{}.post_ffw_norm.weight", i);

            let layer_base = i * 4 * (block_size as usize);
            copy_tensor(&attn_name, layer_base);
            copy_tensor(&ffn_name, layer_base + (block_size as usize));
            copy_tensor(&post_attn_name, layer_base + 2 * (block_size as usize));
            copy_tensor(&post_ffw_name, layer_base + 3 * (block_size as usize));
        }

        // Final Norm
        let final_base = n_layers * 4 * (block_size as usize);
        copy_tensor("output_norm.weight", final_base);

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Norm Bank"),
            contents: &bank,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::spec::{ModelArch, GgufFileType};

    /// Minimal spec for RoPE math tests — no GPU required.
    fn tiny_spec(rope_scale: f32, yarn_alpha: f32, yarn_beta: f32) -> ModelSpec {
        ModelSpec {
            n_vocab: 32,
            n_embd: 8,
            n_layer: 1,
            n_head: 2,
            n_head_kv: 1,
            ff_dim: 16,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale,
            rope_dim: 4,
            yarn_alpha,
            yarn_beta,
            n_ctx: 4,
            head_dim: 4,
            gqa_ratio: 2,
            kv_dim: 4,
            arch: ModelArch::Llama,
            file_type: GgufFileType::Q4_0,
            model_name: "test".to_string(),
            temp_buffer_size: 64,
            kv_cache_size_per_layer: 64,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        }
    }

    /// At distance 0 every (cos, sin) entry must be (1.0, 0.0) regardless of scaling.
    #[test]
    fn rope_table_distance_zero_is_one_zero() {
        let spec = tiny_spec(1.0, 0.0, 0.0);
        let table = PreflightResources::compute_rope_table(&spec, 1.0);
        let n_pairs = spec.rope_dim / 2; // 2
        for p in 0..n_pairs {
            let cos_val = table[p * 2];
            let sin_val = table[p * 2 + 1];
            assert!(
                (cos_val - 1.0).abs() < 1e-6,
                "cos at d=0 p={p} should be 1.0, got {cos_val}"
            );
            assert!(
                sin_val.abs() < 1e-6,
                "sin at d=0 p={p} should be 0.0, got {sin_val}"
            );
        }
    }

    /// Linear scaling (no YaRN): effective theta = base_theta * scale.
    /// At d=1 the angle for pair 0 is theta_0 * scale; verify cos matches.
    #[test]
    fn rope_table_linear_scale_matches_direct_formula() {
        let scale = 0.5_f32;
        let spec = tiny_spec(scale, 0.0, 0.0);
        let table = PreflightResources::compute_rope_table(&spec, scale);
        // theta_0 = 1 / base^(0 / dim) = 1 / 10000^0 = 1.0; effective = 1.0 * 0.5 = 0.5
        let expected_cos = 0.5_f32.cos();
        let n_pairs = spec.rope_dim / 2;
        let cos_val = table[1 * n_pairs * 2 + 0 * 2]; // d=1, p=0, cos
        assert!(
            (cos_val - expected_cos).abs() < 1e-6,
            "linear scale cos mismatch: expected {expected_cos}, got {cos_val}"
        );
    }

    /// YaRN ramp: at the extreme high-frequency end (very short wavelength, i=0 in a small base),
    /// ramp should clamp to 1.0 — meaning no frequency scaling applied to that dimension.
    #[test]
    fn yarn_ramp_high_freq_dim_is_unscaled() {
        // Derivation: base=1.0001 gives theta_0≈1, lambda_0≈2π≈6.28.
        // l_train is n_ctx*scale = 4000*0.5 = 2000.
        // The clamp argument evaluates to ~353, clamping to 1.0 (high-freq: no scaling).
        // Without YaRN the effective theta is theta*scale = theta*0.5.
        // So at d=1: YaRN cos(1.0) ≠ linear cos(0.5).
        let mut spec = tiny_spec(0.5, 0.1, 1.0); // alpha=0.1 > 0 enables YaRN
        spec.n_ctx = 4000;
        spec.rope_base = 1.0001; // tiny base → theta_0 ≈ 1, tiny lambda → ramp clamps to 1.0

        let table_yarn = PreflightResources::compute_rope_table(&spec, 0.5);

        let mut spec_no_yarn = spec.clone();
        spec_no_yarn.yarn_alpha = 0.0; // disables YaRN → falls through to linear
        spec_no_yarn.yarn_beta = 0.0;
        let table_linear = PreflightResources::compute_rope_table(&spec_no_yarn, 0.5);

        let n_pairs = spec.rope_dim / 2;
        // At high-freq dim p=0, d=1: YaRN keeps theta unscaled (cos(1.0)),
        // linear halves it (cos(0.5)). They must differ.
        let yarn_cos = table_yarn[1 * n_pairs * 2 + 0 * 2];
        let linear_cos = table_linear[1 * n_pairs * 2 + 0 * 2];
        assert!(
            (yarn_cos - linear_cos).abs() > 1e-3,
            "YaRN high-freq dim should differ from linear scaling: yarn_cos={yarn_cos:.6}, linear_cos={linear_cos:.6}"
        );
    }
}
