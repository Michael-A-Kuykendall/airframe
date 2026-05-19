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
                // ramp: 1.0 when lambda is short (high-freq), 0.0 when lambda is long (low-freq).
                // Per YaRN paper: high-freq dims need NO scaling (already fine at short distances);
                // low-freq dims need full linear scaling to cover extended distances.
                // Use (1-ramp) to drive the scaling: low-freq (ramp→0) gets full scale, high-freq (ramp→1) gets none.
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

    fn upload_rope_table(device: &wgpu::Device, data: &[f32], scale: f32, spec: &ModelSpec) -> wgpu::Buffer {
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
            contents: bytemuck::cast_slice(data),
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
        // 1. attn_norm (per layer)
        // 2. ffn_norm (per layer)
        // 3. output_norm (final)

        let dim = spec.n_embd;
        let n_layers = spec.n_layer;

        // Block size = dim * 4 bytes
        let block_size = dim * 4;

        // Layout:
        // Layers: [AttnNorm, FfnNorm] interleaved
        // End: [OutputNorm]
        let total_size = (n_layers * 2 + 1) * (block_size as usize);

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

            let layer_base = i * 2 * (block_size as usize);
            copy_tensor(&attn_name, layer_base);
            copy_tensor(&ffn_name, layer_base + (block_size as usize));
        }

        // Final Norm
        let final_base = n_layers * 2 * (block_size as usize);
        copy_tensor("output_norm.weight", final_base);

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Norm Bank"),
            contents: &bank,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
}
