//! GPU-accelerated vision model (SigLIP ViT + Perceiver Resampler).
//!
//! [`GpuVisionModel`] loads `mmproj-model-f16.gguf`, uploads the entire file to a
//! single GPU buffer (bindless pattern), pre-computes byte offsets for every tensor,
//! and exposes [`GpuVisionModel::encode_image`] to convert patch embeddings into
//! 64 visual token embeddings `[64 × 3584]` ready to inject into the LLM token stream.
//!
//! # Memory budget (approx)
//! | Buffer            | Size      |
//! |-------------------|-----------|
//! | vit_blob          | ~1.04 GB  |
//! | activations       | 4.5 MB    |
//! | temp_kqv          | 18 MB     |
//! | temp_ffn          | 17 MB     |
//! | query_state       | 0.9 MB    |
//! | kv_state (3-slot) | 42 MB     |

use std::collections::HashMap;
use std::path::Path;

use wgpu::util::DeviceExt;

use crate::backend::bindless::pipeline::resampler_gpu::{
    ResamplerOffsets, ResamplerParams, ResamplerPipeline,
};
use crate::backend::bindless::pipeline::vit_layer::{
    VitBlockOffsets, VitParams, VitPipeline,
};
use crate::core::error::{LibshimmyError, Result};
use crate::core::f16::f16_bits_to_f32;
use crate::core::image_preproc::{patch_embed_with_pos, normalize_hwc_u8_to_chw_f32, tile_image_chw, MAX_SLICES};
use crate::core::model::{GgufTensorInfo, load_mmproj_gguf_raw};

// ─── Fixed model dimensions (SigLIP-So400M + MiniCPM-V-2.6 Resampler) ────────

const N_VIT_LAYERS:  usize = 27;
const VIT_HIDDEN:    u32   = 1152;
const VIT_N_HEADS:   u32   = 16;
const VIT_HEAD_DIM:  u32   = 72;
const VIT_MLP_DIM:   u32   = 4304;
const VIT_N_TOKENS:  u32   = 1024;
const VIT_LN_EPS:    f32   = 1e-6;

const RSP_N_QUERIES: u32   = 64;
const RSP_D_MODEL:   u32   = 3584;
const RSP_KV_DIM:    u32   = 1152;
const RSP_N_HEADS:   u32   = 16;
const RSP_HEAD_DIM:  u32   = 224;
const RSP_LN_EPS:    f32   = 1e-6;

// ─── GpuVisionModel ──────────────────────────────────────────────────────────

/// GPU-resident SigLIP ViT + Perceiver Resampler.
pub struct GpuVisionModel {
    // ── GPU blob ───────────────────────────────────────────────────────────
    /// Entire mmproj GGUF file uploaded as a flat `array<u32>` storage buffer.
    vit_blob_buf: wgpu::Buffer,

    // ── Compiled pipelines ────────────────────────────────────────────────
    vit_pipeline:        VitPipeline,
    resampler_pipeline:  ResamplerPipeline,

    // ── Pre-computed tensor byte offsets ──────────────────────────────────
    /// One `VitBlockOffsets` per ViT block (27 total).
    layer_offsets:     Vec<VitBlockOffsets>,
    /// `ln1_w` / `ln1_b` point to the ViT post-normalisation weights.
    post_ln_offsets:   VitBlockOffsets,
    vit_params:        VitParams,
    resampler_offsets: ResamplerOffsets,
    resampler_params:  ResamplerParams,

    // ── Persistent working buffers (allocated once, reused per image) ─────
    /// `[VIT_N_TOKENS × VIT_HIDDEN]` f32 — ViT residual stream.
    /// Reused as `vit_features` (read-only) input to the Resampler.
    activations_buf:   wgpu::Buffer,
    /// `[VIT_N_TOKENS × VIT_HIDDEN × 4]` f32 — ViT attention scratch.
    temp_kqv_buf:      wgpu::Buffer,
    /// `[VIT_N_TOKENS × VIT_MLP_DIM]` f32 — ViT FFN scratch.
    temp_ffn_buf:      wgpu::Buffer,
    /// `[RSP_N_QUERIES × RSP_D_MODEL]` f32 — Resampler query state.
    query_state_buf:   wgpu::Buffer,
    /// `[VIT_N_TOKENS × RSP_D_MODEL × 3]` f32 — Resampler KV scratch (3 slots).
    kv_state_buf:      wgpu::Buffer,

    // ── CPU patch embedding weights (small; kept resident for preprocessing) ──
    /// `v.patch_embd.weight` dequantised to f32 — `[HIDDEN × 3 × 14 × 14]`.
    patch_w:   Vec<f32>,
    /// `v.patch_embd.bias` — `[HIDDEN]` f32.
    patch_b:   Vec<f32>,
    /// `v.position_embd.weight` dequantised to f32 — `[4900 × HIDDEN]`.
    pos_embed: Vec<f32>,
}

impl GpuVisionModel {
    /// Load `mmproj-model-f16.gguf` and initialise all GPU resources.
    ///
    /// Blocks until the file is uploaded; use from the model-loading path,
    /// not the hot inference loop.
    pub fn from_mmproj_gguf(
        path: impl AsRef<Path>,
        device: &wgpu::Device,
    ) -> Result<Self> {
        let path = path.as_ref();

        // ── Parse header + tensor index (CPU) ────────────────────────────
        let (tensor_infos, mmap, base_offset) = load_mmproj_gguf_raw(path)?;

        // Build name → absolute byte offset map
        let offsets: HashMap<String, u64> = tensor_infos
            .iter()
            .map(|ti| (ti.name.clone(), base_offset + ti.offset))
            .collect();

        // Build name → tensor info map (for CPU weight loading)
        let info_map: HashMap<&str, &GgufTensorInfo> = tensor_infos
            .iter()
            .map(|ti| (ti.name.as_str(), ti))
            .collect();

        let get = |name: &str| -> Result<u32> {
            let abs = *offsets.get(name).ok_or_else(|| LibshimmyError::WeightMissing {
                weight_id: name.to_string(),
            })?;
            u32::try_from(abs).map_err(|_| LibshimmyError::FixtureError {
                msg: format!("Tensor '{}' byte offset {} exceeds u32 range", name, abs),
            })
        };

        // ── Helpers for loading CPU tensors from mmap ─────────────────────
        let load_f16_tensor = |name: &str| -> Result<Vec<f32>> {
            let ti = *info_map.get(name).ok_or_else(|| LibshimmyError::WeightMissing {
                weight_id: name.to_string(),
            })?;
            let n: usize = ti.dimensions.iter().product();
            let byte_off = (base_offset + ti.offset) as usize;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = byte_off + i * 2;
                let bits = u16::from_le_bytes([mmap[b], mmap[b + 1]]);
                out.push(f16_bits_to_f32(bits));
            }
            Ok(out)
        };

        let load_f32_tensor = |name: &str| -> Result<Vec<f32>> {
            let ti = *info_map.get(name).ok_or_else(|| LibshimmyError::WeightMissing {
                weight_id: name.to_string(),
            })?;
            let n: usize = ti.dimensions.iter().product();
            let byte_off = (base_offset + ti.offset) as usize;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = byte_off + i * 4;
                let bits = u32::from_le_bytes([mmap[b], mmap[b + 1], mmap[b + 2], mmap[b + 3]]);
                out.push(f32::from_bits(bits));
            }
            Ok(out)
        };

        // ── Load CPU patch embedding + positional embedding weights ───────
        let patch_w   = load_f16_tensor("v.patch_embd.weight")?;
        let patch_b   = load_f32_tensor("v.patch_embd.bias")?;
        let pos_embed = load_f16_tensor("v.position_embd.weight")?;

        // ── Upload entire GGUF file to GPU ────────────────────────────────
        let vit_blob_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("VitBlob"),
            contents: &mmap[..],
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // ── Build per-layer VitBlockOffsets (27 layers) ───────────────────
        let mut layer_offsets: Vec<VitBlockOffsets> = Vec::with_capacity(N_VIT_LAYERS);
        for i in 0..N_VIT_LAYERS {
            let p = |suffix: &str| get(&format!("v.blk.{}.{}", i, suffix));
            layer_offsets.push(VitBlockOffsets {
                ln1_w:    p("ln1.weight")?,
                ln1_b:    p("ln1.bias")?,
                attn_q_w: p("attn_q.weight")?,
                attn_q_b: p("attn_q.bias")?,
                attn_k_w: p("attn_k.weight")?,
                attn_k_b: p("attn_k.bias")?,
                attn_v_w: p("attn_v.weight")?,
                attn_v_b: p("attn_v.bias")?,
                attn_o_w: p("attn_out.weight")?,
                attn_o_b: p("attn_out.bias")?,
                ln2_w:    p("ln2.weight")?,
                ln2_b:    p("ln2.bias")?,
                ffn_up_w: p("ffn_up.weight")?,
                ffn_up_b: p("ffn_up.bias")?,
                ffn_dn_w: p("ffn_down.weight")?,
                ffn_dn_b: p("ffn_down.bias")?,
            });
        }

        // ── Build post-LN offsets (ln1_w/ln1_b repurposed for post_ln shader) ──
        let post_ln_offsets = VitBlockOffsets {
            ln1_w:    get("v.post_ln.weight")?,
            ln1_b:    get("v.post_ln.bias")?,
            // Remaining fields are unused by main_vit_post_ln kernel — zero fill.
            ..bytemuck::Zeroable::zeroed()
        };

        // ── Build ResamplerOffsets ────────────────────────────────────────
        let resampler_offsets = ResamplerOffsets {
            query_embeds: get("resampler.query")?,
            kv_weight:    get("resampler.kv.weight")?,
            ln_q_w:       get("resampler.ln_q.weight")?,
            ln_q_b:       get("resampler.ln_q.bias")?,
            ln_kv_w:      get("resampler.ln_kv.weight")?,
            ln_kv_b:      get("resampler.ln_kv.bias")?,
            attn_q_w:     get("resampler.attn.q.weight")?,
            attn_q_b:     get("resampler.attn.q.bias")?,
            attn_k_w:     get("resampler.attn.k.weight")?,
            attn_k_b:     get("resampler.attn.k.bias")?,
            attn_v_w:     get("resampler.attn.v.weight")?,
            attn_v_b:     get("resampler.attn.v.bias")?,
            attn_out_w:   get("resampler.attn.out.weight")?,
            attn_out_b:   get("resampler.attn.out.bias")?,
            pos_embed_k:  get("resampler.pos_embed_k")?,
            ln_post_w:    get("resampler.ln_post.weight")?,
            ln_post_b:    get("resampler.ln_post.bias")?,
            proj_w:       get("resampler.proj.weight")?,
            pad0: 0, pad1: 0,
        };

        // ── Params structs ────────────────────────────────────────────────
        let vit_params = VitParams {
            hidden_dim: VIT_HIDDEN,
            n_heads:    VIT_N_HEADS,
            head_dim:   VIT_HEAD_DIM,
            mlp_dim:    VIT_MLP_DIM,
            n_tokens:   VIT_N_TOKENS,
            ln_eps:     VIT_LN_EPS,
            pad0: 0, pad1: 0,
        };

        let resampler_params = ResamplerParams {
            n_queries: RSP_N_QUERIES,
            n_vit:     VIT_N_TOKENS,
            d_model:   RSP_D_MODEL,
            kv_dim:    RSP_KV_DIM,
            n_heads:   RSP_N_HEADS,
            head_dim:  RSP_HEAD_DIM,
            ln_eps:    RSP_LN_EPS,
            pad0: 0,
        };

        // ── Compile pipelines ─────────────────────────────────────────────
        let vit_pipeline       = VitPipeline::new(device);
        let resampler_pipeline = ResamplerPipeline::new(device);

        // ── Allocate persistent working buffers ───────────────────────────
        let storage = wgpu::BufferUsages::STORAGE;
        let storage_copy = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

        let activations_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:             Some("VitActivations"),
            size:              (VIT_N_TOKENS * VIT_HIDDEN) as u64 * 4,
            usage:             storage_copy,
            mapped_at_creation: false,
        });

        let temp_kqv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:             Some("VitTempKQV"),
            size:              (VIT_N_TOKENS * VIT_HIDDEN * 4) as u64 * 4,
            usage:             storage,
            mapped_at_creation: false,
        });

        let temp_ffn_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:             Some("VitTempFFN"),
            size:              (VIT_N_TOKENS * VIT_MLP_DIM) as u64 * 4,
            usage:             storage,
            mapped_at_creation: false,
        });

        let query_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:             Some("RspQueryState"),
            size:              (RSP_N_QUERIES * RSP_D_MODEL) as u64 * 4,
            usage:             storage,
            mapped_at_creation: false,
        });

        let kv_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:             Some("RspKVState"),
            size:              (VIT_N_TOKENS * RSP_D_MODEL * 3) as u64 * 4,
            usage:             storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        println!(
            "✅ GpuVisionModel ready: {} ViT layers, blob {:.1} MB",
            N_VIT_LAYERS,
            mmap.len() as f64 / 1_048_576.0,
        );

        Ok(Self {
            vit_blob_buf,
            vit_pipeline,
            resampler_pipeline,
            layer_offsets,
            post_ln_offsets,
            vit_params,
            resampler_offsets,
            resampler_params,
            activations_buf,
            temp_kqv_buf,
            temp_ffn_buf,
            query_state_buf,
            kv_state_buf,
            patch_w,
            patch_b,
            pos_embed,
        })
    }

    /// Encode patch embeddings through the ViT + Resampler on the GPU.
    ///
    /// # Arguments
    /// * `patch_embeddings` – `[VIT_N_TOKENS × VIT_HIDDEN]` f32 slice produced
    ///   by the CPU patch-embedding step (patchify + project + add positional embed).
    ///
    /// # Returns
    /// `[RSP_N_QUERIES × RSP_D_MODEL]` = `[64 × 3584]` f32 visual token embeddings,
    /// ready to be spliced into the LLM token stream at the `<image>` position.
    pub fn encode_image(
        &self,
        patch_embeddings: &[f32],
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
    ) -> Vec<f32> {
        assert_eq!(
            patch_embeddings.len(),
            (VIT_N_TOKENS * VIT_HIDDEN) as usize,
            "patch_embeddings length mismatch: expected {} got {}",
            VIT_N_TOKENS * VIT_HIDDEN,
            patch_embeddings.len(),
        );

        // ── 1. Upload patch embeddings to the activations buffer ──────────
        queue.write_buffer(
            &self.activations_buf,
            0,
            bytemuck::cast_slice(patch_embeddings),
        );

        // ── 2. Run 27 ViT transformer blocks ─────────────────────────────
        for i in 0..N_VIT_LAYERS {
            self.vit_pipeline.run_vit_block(
                device, queue,
                &self.vit_blob_buf,
                &self.activations_buf,
                &self.temp_kqv_buf,
                &self.temp_ffn_buf,
                self.layer_offsets[i],
                self.vit_params,
            );
        }

        // ── 3. Post-ViT LayerNorm ─────────────────────────────────────────
        self.vit_pipeline.run_post_ln(
            device, queue,
            &self.vit_blob_buf,
            &self.activations_buf,
            &self.temp_kqv_buf,
            &self.temp_ffn_buf,
            self.post_ln_offsets,
            self.vit_params,
        );

        // ── 4. Perceiver Resampler ────────────────────────────────────────
        // activations_buf now holds ViT output [1024 × 1152]; pass as vit_features.
        self.resampler_pipeline.run_resampler(
            device, queue,
            &self.vit_blob_buf,
            &self.activations_buf,   // ← vit_features (read-only in resampler)
            &self.query_state_buf,
            &self.kv_state_buf,
            self.resampler_offsets,
            self.resampler_params,
        );

        // ── 5. Read back output: kv_state_buf[0..n_queries*d_model] ───────
        let output_bytes = (RSP_N_QUERIES * RSP_D_MODEL) as u64 * 4;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label:             Some("VisionOutput Staging"),
            size:              output_bytes,
            usage:             wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("VisionReadback") });
        enc.copy_buffer_to_buffer(&self.kv_state_buf, 0, &staging, 0, output_bytes);
        queue.submit(Some(enc.finish()));

        readback_f32(device, &staging)
    }

    /// End-to-end pipeline: raw RGB image → visual token embeddings.
    ///
    /// Tiles the image (up to `MAX_SLICES` slices + 1 thumbnail), runs the
    /// full ViT + Resampler pipeline on each tile, and returns one
    /// `[64 × 3584]` f32 block per tile.
    ///
    /// # Arguments
    /// * `pixels_hwc` – packed `[H × W × 3]` u8 bytes in HWC / RGB order.
    /// * `h`, `w`     – image height and width in pixels.
    ///
    /// # Returns
    /// `Vec` of `N_tiles` vectors; each inner `Vec<f32>` has length
    /// `64 × 3584 = 229 376`.
    pub fn encode_tiles(
        &self,
        pixels_hwc: &[u8],
        h: usize,
        w: usize,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
    ) -> Vec<Vec<f32>> {
        let chw   = normalize_hwc_u8_to_chw_f32(pixels_hwc, h, w);
        let tiles = tile_image_chw(&chw, h, w, MAX_SLICES);

        tiles.iter().map(|tile| {
            let patch_embeds = patch_embed_with_pos(
                tile,
                &self.patch_w,
                &self.patch_b,
                &self.pos_embed,
            );
            self.encode_image(&patch_embeds, device, queue)
        }).collect()
    }
}

// ─── Readback helper ─────────────────────────────────────────────────────────

/// Map a staging buffer and copy its contents to a `Vec<f32>`.
/// Blocks until the GPU has completed pending work on this buffer.
fn readback_f32(device: &wgpu::Device, staging: &wgpu::Buffer) -> Vec<f32> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

    loop {
        device.poll(wgpu::PollType::Poll).expect("GPU device lost during vision readback");
        if let Ok(res) = rx.try_recv() {
            res.expect("Vision output buffer map failed");
            break;
        }
    }

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
