use crate::backend::bindless::loader::BindlessModel;
use crate::core::spec::ModelSpec;
use crate::core::tensor::Tensor;
use wgpu::util::DeviceExt;

/// The "Pre-Flight" deck.
/// Contains resources that are "Fused" (Pre-computed/Extracted)
/// from the raw model to prepare for high-speed GPU execution.
pub struct PreflightResources {
    /// Pre-computed RoPE frequencies (Cos/Sin pairs).
    /// Layout: [Layer 0..N] -> (Though RoPE is usually shared)
    /// Actually simplified: [HeadDim/2] complex pairs repeated.
    pub rope_cache_buffer: wgpu::Buffer,

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

        let rope_buffer = Self::build_rope_cache(device, spec);
        let norm_buffer = Self::build_norm_bank_from_ram(device, raw_data, metadata, spec);

        Self {
            rope_cache_buffer: rope_buffer,
            norm_bank_buffer: norm_buffer,
        }
    }

    fn build_rope_cache(device: &wgpu::Device, spec: &ModelSpec) -> wgpu::Buffer {
        // Pre-compute full cos/sin lookup table using YaRN per-dimension scaling.
        //
        // YaRN (Peng et al. 2023) applies a non-uniform frequency correction per
        // RoPE dimension pair:
        //
        //   For pair i with base theta_i = 1 / base^(2i/D):
        //     wavelength lambda_i = 2*pi / theta_i = 2*pi * base^(2i/D)
        //     low  = L_train / yarn_alpha  (long-wavelength dims: no scaling needed)
        //     high = L_train / yarn_beta   (short-wavelength dims: full linear scaling)
        //
        //   ramp_i = clamp((L_train / lambda_i - yarn_alpha) / (yarn_beta - yarn_alpha), 0.0, 1.0)
        //   effective_theta_i = theta_i * (ramp_i / rope_scale + (1.0 - ramp_i))
        //     = theta_i when ramp=0 (low freq, no scaling)
        //     = theta_i / rope_scale when ramp=1 (high freq, full linear equiv)
        //
        // Layout: [distance][pair] each entry = (cos, sin).
        //   table[d * n_pairs * 2 + p * 2 + 0] = cos(d * effective_theta_p)
        //   table[d * n_pairs * 2 + p * 2 + 1] = sin(d * effective_theta_p)
        let dim = spec.rope_dim;
        let base = spec.rope_base;
        let scale = spec.rope_scale; // < 1.0 extends context: rope_scale = L_train / L_target
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
                    return theta * scale; // plain linear scaling
                }
                let lambda = std::f32::consts::TAU / theta; // wavelength = 2*pi / theta
                let ramp = ((l_train as f32 / lambda - alpha) / (beta - alpha)).clamp(0.0, 1.0);
                // ramp=0: low-freq dim, no scaling (theta unchanged)
                // ramp=1: high-freq dim, full linear scaling (theta * scale)
                theta * (ramp * scale + (1.0 - ramp))
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

        let scaling_mode = if use_yarn { "YaRN" } else { "linear" };
        println!(
            "[Preflight] RoPE Lookup Table: {}×{} = {} entries ({:.1} KB, Base={}, Dim={}, Scale={}, Mode={})",
            max_dist, n_pairs, table_len,
            (table_len * 4) as f64 / 1024.0,
            base, dim, scale, scaling_mode
        );

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RoPE Cos/Sin Lookup Table"),
            contents: bytemuck::cast_slice(&table),
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
