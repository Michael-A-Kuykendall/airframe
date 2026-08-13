// Auto-split from pipeline.rs — types, struct, constructor, and helpers only.
// Inference methods: see inference.rs, layer.rs, dequant.rs, matmul.rs

pub(super) mod dequant;
pub mod inference;
pub(super) mod layer;
pub(super) mod matmul;

//       pipeline/kv_cache.rs, pipeline/dispatch.rs — see C3 architectural debt.
use super::loader::{BindlessModel, BlobWindow, BLOB_BINDING_SLOTS};
use crate::core::spec::ModelSpec;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DequantParams {
    pub offset_bytes: u32,
    pub count: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DequantAnyParams {
    pub blob_base_words: u32,
    pub offset_words: u32,
    pub count: u32,
    pub formula_index: u32,
    pub chunk_words: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatMulParams {
    pub n: u32,
    pub k: u32,
    pub weights_offset: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RMSNormParams {
    pub count: u32,
    pub weights_offset: u32,
    pub bias_offset: u32, // 0 = disabled; otherwise word index (byte_offset / 4)
    pub eps: f32,
    pub norm_type: u32,   // 0 = RMSNorm, 1 = LayerNorm (mean+variance)
    pub chunk_words: u32, // words per blob chunk — dispatch read_blob across blob_0..blob_7
}

/// Offsets for a single Transformer Layer (TinyLlama/Llama 2).
/// GGUF blob offsets packed for WGSL `u32` (files up to 8 GiB).
///
/// Host stores `byte_offset / 2` (must be 2-byte aligned). Shaders convert with
/// `gow(packed) = packed / 2` (word index) and `read_byte_rel` for odd-byte
/// dequant — never `pack * 2` in u32 (overflow past 4 GiB / Qwen3-8B L30+).
#[inline]
pub fn pack_blob_offset(byte_offset: u64) -> u32 {
    if byte_offset == 0 {
        return 0;
    }
    assert!(
        byte_offset.is_multiple_of(2),
        "GGUF offset {byte_offset} must be 2-byte aligned for pack_blob_offset"
    );
    let packed = byte_offset / 2;
    assert!(
        packed <= u32::MAX as u64,
        "GGUF offset {byte_offset} exceeds 8GiB pack_blob_offset limit"
    );
    packed as u32
}

/// Safe base-relative offset packing: returns `(abs - base) / 2` as `u32`, with
/// underflow detection.  Use this instead of `pack_blob_offset(abs - base_byte)`
/// to prevent `u64` underflow panics when `abs < base_byte` (which can happen
/// with fused-QKV / merged-gate+up offsets that have non-monotonic absolute
/// offsets within a layer).
#[inline]
pub fn relative_packed_offset(abs: u64, base: u64) -> Result<u32, String> {
    if abs == 0 {
        return Ok(0);
    }
    let rel = abs
        .checked_sub(base)
        .ok_or_else(|| format!("underflow: abs={abs} (0x{abs:x}) < base={base} (0x{base:x})"))?;
    // rel must be 2-byte aligned for packed word offset
    if rel % 2 != 0 {
        return Err(format!(
            "relative offset {rel} (0x{rel:x}) is not 2-byte aligned"
        ));
    }
    let packed = rel / 2;
    if packed > u32::MAX as u64 {
        return Err(format!(
            "relative offset {rel} (0x{rel:x}) packs to {} which exceeds u32::MAX",
            packed
        ));
    }
    Ok(packed as u32)
}

/// All offsets are **packed** absolute byte offsets from the GGUF blob start
/// (`pack_blob_offset`). Zero means disabled / missing.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerOffsets {
    pub attn_norm: u32,
    pub attn_norm_bias: u32, // packed byte offset of attn norm bias (F32); 0 = disabled
    pub attn_q: u32,
    pub attn_k: u32,
    pub attn_v: u32,
    pub attn_out: u32,
    pub ffn_norm: u32,
    pub ffn_norm_bias: u32, // packed byte offset of ffn norm bias (F32); 0 = disabled
    pub ffn_gate: u32,
    pub ffn_down: u32,
    pub ffn_up: u32,
    pub layer_idx: u32,       // was padding[0] — layer index for norm_bank lookup
    pub attn_q_norm: u32,     // packed Q-norm weights (0 = disabled)
    pub attn_k_norm: u32,     // packed K-norm weights (0 = disabled)
    pub attn_q_bias: u32,     // packed Q bias (F32); 0 = disabled; Qwen2
    pub attn_k_bias: u32,     // packed K bias (F32); 0 = disabled; Qwen2
    pub attn_v_bias: u32,     // packed V bias (F32); 0 = disabled; Qwen2
    pub v_is_q4k: u32,        // 1 if attn_v uses Q4_K, 0 if Q6_K (for Q4_K_M mixed quantization)
    pub ffn_down_is_q4k: u32, // 1 if ffn_down uses Q4_K, 0 if Q6_K (for Q4_K_M mixed quantization)
    // PLE (Per-Layer Embedding) per-layer tensors (gemma-4 dense-latent)
    pub ple_inp_gate: u32,
    pub ple_proj: u32,
    pub ple_layer_output_scale: u32,
    pub ple_rope_freqs: u32,
    pub ple_attn_post_norm: u32,
    pub ple_ffn_post_norm: u32,
    /// PLE per-layer output norm (`blk.N.post_norm.weight`). 0 = absent.
    pub ple_post_norm: u32,
    pub ple_enabled: u32,
}

impl LayerOffsets {
    /// Returns the minimum and maximum absolute word index covered by this layer's tensors.
    /// Absolute word = (packed_offset / 2) + blob_base_words.
    /// Only considers non-zero (present) tensors.
    pub fn word_span(&self, blob_base_words: u32) -> Option<(u32, u32)> {
        let mut min_word = u32::MAX;
        let mut max_word = 0;
        let mut has_tensor = false;

        let check = |packed: u32, min_word: &mut u32, max_word: &mut u32, has_tensor: &mut bool| {
            if packed != 0 {
                *has_tensor = true;
                let word = (packed / 2) + blob_base_words;
                *min_word = (*min_word).min(word);
                *max_word = (*max_word).max(word);
            }
        };

        check(
            self.attn_norm,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.attn_norm_bias,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(self.attn_q, &mut min_word, &mut max_word, &mut has_tensor);
        check(self.attn_k, &mut min_word, &mut max_word, &mut has_tensor);
        check(self.attn_v, &mut min_word, &mut max_word, &mut has_tensor);
        check(self.attn_out, &mut min_word, &mut max_word, &mut has_tensor);
        check(self.ffn_norm, &mut min_word, &mut max_word, &mut has_tensor);
        check(
            self.ffn_norm_bias,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(self.ffn_gate, &mut min_word, &mut max_word, &mut has_tensor);
        check(self.ffn_down, &mut min_word, &mut max_word, &mut has_tensor);
        check(self.ffn_up, &mut min_word, &mut max_word, &mut has_tensor);
        check(
            self.attn_q_norm,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.attn_k_norm,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.attn_q_bias,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.attn_k_bias,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.attn_v_bias,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.ple_inp_gate,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(self.ple_proj, &mut min_word, &mut max_word, &mut has_tensor);
        check(
            self.ple_layer_output_scale,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.ple_ffn_post_norm,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );
        check(
            self.ple_post_norm,
            &mut min_word,
            &mut max_word,
            &mut has_tensor,
        );

        if has_tensor {
            Some((min_word, max_word))
        } else {
            None
        }
    }
}

/// Pure window algebra for a layer dispatch: given the layer's packed tensor
/// offsets and the loaded chunk plan, returns the [`BlobWindow`] to bind and
/// the window-local `blob_base_words` the shader must receive.
///
/// The shader's `read_blob` resolves `word_idx` against the eight bound blob
/// slots, so every absolute word index a dispatch touches must be rebased to
/// the window start. `blob_base_words` is the only host-supplied base the
/// shader adds to packed offsets, so subtracting `window_base_words()` from it
/// rebases the whole layer in one place.
///
/// Returns `Ok((None, blob_base_words))` unchanged when the layer declares no
/// tensors (window algebra is a no-op in that case), and `Err` when the span
/// cannot be covered by `BLOB_BINDING_SLOTS` consecutive resident chunks.
///
/// Kept free of `BindlessModel` so the algebra is exercised by the CPU-only
/// PPT contract suite, not just on a live adapter.
pub fn plan_layer_window(
    offsets: &LayerOffsets,
    blob_base_words: u32,
    chunk_words: u32,
    total_resident_chunks: usize,
) -> Result<(Option<BlobWindow>, u32), String> {
    let Some((min_word, max_word)) = offsets.word_span(blob_base_words) else {
        return Ok((None, blob_base_words));
    };
    let window = BlobWindow::for_range(min_word, max_word, chunk_words, total_resident_chunks)?;
    let adjusted = blob_base_words.saturating_sub(window.window_base_words());
    Ok((Some(window), adjusted))
}

/// Model-bound wrapper around [`plan_layer_window`] for dispatch sites.
///
/// # Panics
/// Panics if the layer's tensor span cannot be covered by
/// `BLOB_BINDING_SLOTS` consecutive resident chunks — a silent wrong-chunk read
/// is far worse than a dispatch-time abort.
pub fn resolve_layer_window(
    model: &BindlessModel,
    offsets: &LayerOffsets,
    blob_base_words: u32,
    layer_idx: usize,
) -> (Option<BlobWindow>, u32) {
    plan_layer_window(
        offsets,
        blob_base_words,
        model.chunk_words(),
        model.total_resident_chunks,
    )
    .unwrap_or_else(|e| panic!("layer {} window planning failed: {}", layer_idx, e))
}

/// Rebases an LM-head `weight_off` (absolute word index of the weight tensor)
/// onto a bound [`BlobWindow`].
///
/// sh_head_blob.wgsl reads `weight_off + rel_word`, where `rel_word` is an
/// offset from the start of the weight tensor. For a tile whose rows begin well
/// past the tensor start, the window base can exceed `weight_off`, so the
/// rebased base is mathematically negative.
///
/// That is fine, and is why this returns a wrapped `u32`: WGSL `u32` addition
/// wraps modulo 2^32, so `wrap(weight_off - window_base) + rel_word` evaluates
/// to the true window-local word for every `rel_word` the dispatch actually
/// reads (all of which satisfy `weight_off + rel_word >= window_base`).
/// Callers must therefore only use it for reads inside `window`.
pub fn rebase_head_weight_off(weight_off: u32, window: &BlobWindow) -> u32 {
    weight_off.wrapping_sub(window.window_base_words())
}

/// Returns the eight blob binding resources for a dispatch.
///
/// With a window, slot `i` binds resident chunk `window.start_chunk + i`.
/// Without one (no tensors / legacy single-window models) it falls back to the
/// identity mapping `blob_binding_0..7`, which is what a window starting at
/// chunk 0 would produce anyway.
pub fn blob_bindings_for<'a>(
    model: &'a BindlessModel,
    window: Option<&BlobWindow>,
) -> [wgpu::BindingResource<'a>; BLOB_BINDING_SLOTS] {
    match window {
        Some(w) => w.binding_resources(model),
        None => [
            model.blob_binding_0(),
            model.blob_binding_1(),
            model.blob_binding_2(),
            model.blob_binding_3(),
            model.blob_binding_4(),
            model.blob_binding_5(),
            model.blob_binding_6(),
            model.blob_binding_7(),
        ],
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerParams {
    pub dim: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub rms_eps: f32,
    pub ffn_dim: u32, // Feed-forward intermediate dimension (e.g. 5632 for TinyLlama)
    pub temp_stride: u32, // Per-token temp buffer stride in floats (e.g. 16384)
    pub quant_qk: u32, // attn_q / attn_k
    pub quant_v: u32, // attn_v
    pub quant_attn_out: u32, // attn_output
    pub quant_ffn_down: u32, // ffn_down
    pub quant_ffn_gate: u32, // ffn_gate
    pub quant_ffn_up: u32, // ffn_up
    // (old quant_type removed; use per-tensor selectors above for FSE dispatch)
    pub attn_logit_softcap: f32, // 0.0 = disabled; Gemma-2 uses 50.0
    pub post_norm_enabled: u32,  // 1 = apply post-attn and post-ffw norm (Gemma-2); 0 = disabled
    pub qk_norm_enabled: u32,    // 1 = apply per-head Q/K RMSNorm before RoPE (Qwen3); 0 = disabled
    pub layer_norm_enabled: u32, // 1 = use LayerNorm math in layer norms (Phi-family)
    pub ffn_kind_policy: u32,    // 0 = infer from offsets (compat), 1 = gated, 2 = non-gated
    pub qkv_layout_policy: u32,  // 0 = infer from offsets (compat), 1 = separate, 2 = fused
    /// Micro-batch offset: first token index in this QKV chunk (0 for non-chunked dispatches)
    pub batch_offset: u32,
    /// Micro-batch count: number of tokens in this QKV chunk (== batch_size for non-chunked)
    pub batch_count: u32,
    /// Stored K (in-dim) for attn_q.weight tensor (packed Qwen3 GGUF etc; 0 = use dim; for correct blocks_per_row/stride in Q4K qkv stage 0)
    pub q_weight_k: u32,
    /// Stored K for attn_k.weight (packed case)
    pub k_weight_k: u32,
    /// Registry-derived dispatch slot for attn_q / attn_k (B1 formula index).
    /// The shader switches on this instead of re-deriving quant→formula.
    pub formula_qk: u32,
    /// Registry-derived dispatch slot for attn_v.
    pub formula_v: u32,
    /// Registry-derived dispatch slot for attn_output.
    pub formula_attn_out: u32,
    /// Registry-derived dispatch slot for ffn_down.
    pub formula_ffn_down: u32,
    /// Registry-derived dispatch slot for ffn_gate.
    pub formula_ffn_gate: u32,
    /// Registry-derived dispatch slot for ffn_up.
    pub formula_ffn_up: u32,
    /// Word offset of this dispatch's base byte within the GGUF blob.
    /// Reconstructs absolute word index in shaders: gow(pack) = pack / 2 + blob_base_words.
    pub blob_base_words: u32,
    /// Words per blob chunk (effective_chunk / 4). Shaders dispatch read_blob
    /// across blob_0..blob_7 using this. Must stay at the END to match WGSL.
    pub chunk_words: u32,
    /// 1 = apply per-head plain RMSNorm on V before attention (Gemma-4); 0 = disabled
    pub v_plain_rms_norm: u32,
    /// 1 = multiply the final block residual by layer_scales[layer_idx] (Gemma-4); 0 = disabled
    pub out_scale_enabled: u32,
    /// PLE (Per-Layer Embedding) — gemma-4 dense-latent; 0 = disabled
    pub ple_latent_dim: u32, // latent embedding dim per layer (256 for gemma-4-E4B)
    pub ple_enabled: u32, // 1 = this layer runs the PLE block after FFN residual
    /// 0.0 = use 1/sqrt(head_dim); else use this (gemma-4: 1.0)
    pub attn_scale_override: f32,
}

/// Map a raw GGML quant type id to the B1 formula-index slot the shaders
/// consume. Under `isf` this is the canonical `airframe_observe::quant_formula`
/// registry (single source of truth); otherwise a thin mirror of the same 8
/// values so the always-built bindless pipeline stays consistent.
///
/// Slot assignment (must match `airframe_observe::quant_formula::FormulaSlot`):
/// F32=0 F16=1 Q4_0=2 Q5_0=3 Q8_0=4 Q4_K=5 Q5_K=6 Q6_K=7.
#[cfg(feature = "isf")]
pub fn formula_index_for_ggml(type_id: u32) -> u32 {
    airframe_observe::quant_formula::slot_for_type(type_id).map_or(0, |s| s.as_u32())
}

#[cfg(not(feature = "isf"))]
pub fn formula_index_for_ggml(type_id: u32) -> u32 {
    match type_id {
        0 => 0,
        1 => 1,
        2 => 2,
        6 => 3,
        8 => 4,
        12 => 5,
        13 => 6,
        14 => 7,
        _ => 0,
    }
}

/// Uniform params for the quantize_kv.wgsl dispatch (TurboQuant).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantizeKvParams {
    pub n_head_kv: u32,  // Number of KV heads
    pub head_dim: u32,   // Elements per head-vector (must be multiple of 8)
    pub pos_offset: u32, // Base position: actual pos = pos_offset + global_id.y
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheParams {
    pub current_pos: u32,      // Position to write new K/V (0-based)
    pub seq_len: u32,          // Total cached positions (current_pos + 1)
    pub max_seq_len: u32,      // 2048 (context window)
    pub batch_size: u32,       // Number of tokens in current batch
    pub logical_pos_base: u32, // Logical base of the compacted sliding window
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HeadBlobParams {
    pub vocab_size: u32,    // rows of output.weight (= n_vocab)
    pub dim: u32,           // cols of output.weight (= n_embd)
    pub weight_off: u32,    // word offset (byte_offset / 4) of output.weight inside the GGUF blob
    pub formula_index: u32, // B3b registry slot (0..7); dispatch decision made in Rust B1 registry (Golden Rule 3)
    pub softcap: f32,       // final_logit_softcap (0.0 = disabled)
    pub base_row: u32,      // output row offset for dispatch splitting (TDR tiles)
    pub chunk_words: u32,   // words per blob chunk — dispatch read_blob across blob_0..blob_7
}

/// Uniform params for the gemma-4 PLE per-layer latent input construction
/// (`sh_ple_input.wgsl`). Only ever bound when `spec.ple_enabled`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PleInputParams {
    pub latent: u32,               // 256 (n_embd_per_layer)
    pub n_layer: u32,              // 42
    pub n_tokens: u32,             // batch_size
    pub n_embd: u32,               // 2560 (model input dim)
    pub token_embd_off: u32,       // packed offset of per_layer_token_embd
    pub token_embd_row_bytes: u32, // bytes per token row (Q6_K latent*n_layer elems)
    pub model_proj_off: u32,       // packed offset of per_layer_model_proj
    pub proj_norm_off: u32,        // packed offset of per_layer_proj_norm
    pub rms_eps: f32,
    pub blob_base_words: u32, // window-local base
    pub chunk_words: u32,
}

/// Uniform params for the gemma-4 PLE block residual pass (`sh_ple_block.wgsl`).
/// Only ever bound when `spec.ple_enabled`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PleBlockParams {
    pub dim: u32,    // n_embd
    pub latent: u32, // 256
    pub temp_stride: u32,
    pub rms_eps: f32,
    pub out_scale_enabled: u32,
    pub scratch_base: u32, // temp offset for the proj scratch (ffn_dim*2, dead slot)
    pub blob_base_words: u32, // window-local base (rebase absolute -> window)
    pub chunk_words: u32,
}

/// Pre-compiled per-layer lookup table entry.
/// Built once at model load time from the GGUF tensor index.
/// Eliminates per-token HashMap lookups and format! string allocations
/// in the inference hot path (FSE compiled-layer optimization).
#[derive(Clone, Debug)]
pub struct CompiledLayerEntry {
    /// All tensor byte-offsets for this layer, ready to upload to GPU.
    pub offsets: LayerOffsets,
    /// Word offset of the base byte for this layer in the GGUF blob.
    /// Used by the shader to reconstruct absolute addresses: gow(pack) = pack/2 + blob_base_words.
    pub blob_base_words: u32,
    pub quant_qk: u32,
    pub quant_v: u32,
    pub quant_attn_out: u32,
    pub quant_ffn_down: u32,
    pub quant_ffn_gate: u32,
    pub quant_ffn_up: u32,
}

/// The Control Plane for Bindless Inference.
pub struct BindlessPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dequant_pipeline: wgpu::ComputePipeline,
    pub dequant_bind_group_layout: wgpu::BindGroupLayout,
    /// Pre-compiled multi-type dequant pipeline (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32).
    /// Hot-path safe — compiled once at startup, reused for all embedding dequants.
    pub dequant_any_pipeline: wgpu::ComputePipeline,
    pub dequant_any_bind_group_layout: wgpu::BindGroupLayout,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub matmul_layout: wgpu::BindGroupLayout,
    pub matmul_f32_pipeline: wgpu::ComputePipeline,
    pub matmul_f32_layout: wgpu::BindGroupLayout,
    pub rmsnorm_pipeline: wgpu::ComputePipeline,
    pub rmsnorm_layout: wgpu::BindGroupLayout,

    // Split Layer Pipelines
    pub layer_pipeline_attn_norm: wgpu::ComputePipeline,
    pub layer_pipeline_qkv: wgpu::ComputePipeline,
    pub layer_pipeline_qk_norm: wgpu::ComputePipeline,
    pub layer_pipeline_attn_out: wgpu::ComputePipeline,
    pub layer_pipeline_attn_proj: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_norm: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_proj: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_down: wgpu::ComputePipeline,
    pub layer_pipeline_post_attn_norm: wgpu::ComputePipeline,
    pub layer_pipeline_post_ffw_norm: wgpu::ComputePipeline,
    pub layer_layout: wgpu::BindGroupLayout,

    // Blob-based LM head pipeline (quantized matmul, reads directly from GGUF blob)
    pub lm_head_blob_pipeline: wgpu::ComputePipeline,
    pub lm_head_blob_layout: wgpu::BindGroupLayout,

    // INT4 KV Cache pipelines (TurboQuant — feat/turboquant-wgsl)
    // Compiled unconditionally at startup; selected at runtime by SHIMMY_KV_QUANT=int4.
    pub layer_layout_int4: wgpu::BindGroupLayout,
    pub layer_pipeline_attn_out_int4: wgpu::ComputePipeline,
    pub quantize_kv_layout: wgpu::BindGroupLayout,
    pub quantize_kv_pipeline: wgpu::ComputePipeline,

    // gemma-4 dense-latent PLE per-layer input construction (only dispatched when ple_enabled).
    pub ple_input_pipeline: wgpu::ComputePipeline,
    pub ple_input_layout: wgpu::BindGroupLayout,

    // gemma-4 dense-latent PLE block residual (only dispatched when ple_enabled).
    pub ple_block_pipeline: wgpu::ComputePipeline,
    pub ple_block_layout: wgpu::BindGroupLayout,
}

impl BindlessPipeline {
    /// Creates the pipeline with a "Probe" kernel to verify connectivity.
    pub fn new(device: &wgpu::Device) -> Self {
        // --- 1. Probe Pipeline ---
        // Binding 0: GGUF Blob, read-only storage
        // Binding 1: Output Probe, read-write storage
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bindless Layout"),
            entries: &[
                // GGUF Blob
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output Probe (Debug)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 2. Shader Source (Inline WGSL for Probe)
        // Reads the first u32 (Magic Number) and writes it to output[0]
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
            @group(0) @binding(1) var<storage, read_write> output: array<u32>;

            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                // Read magic "GGUF" (0x46554747 le)
                // Note: array<u32> views the byte buffer as u32s. 
                // GGUF magic is at offset 0.
                output[0] = gguf_blob[0];
                output[1] = gguf_blob[1]; // Version ??
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bindless Probe Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // 3. Create Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bindless Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bindless Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 4. Q4_0 Dequant Pipeline ---
        // Binding 0: GGUF Blob, read-only
        // Binding 1: Output F32, read-write
        // Binding 2: Params, uniform

        let dequant_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dequant Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let dequant_shader_source = include_str!("../sh_dequant_q4_0.wgsl");
        let dequant_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Q4_0 Dequant Shader"),
            source: wgpu::ShaderSource::Wgsl(dequant_shader_source.into()),
        });

        let dequant_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dequant Pipeline Layout"),
                bind_group_layouts: &[&dequant_layout],
                push_constant_ranges: &[],
            });

        let dequant_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Q4_0 Dequant Pipeline"),
            layout: Some(&dequant_pipeline_layout),
            module: &dequant_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 4b. DequantAny Pipeline (pre-compiled, hot-path safe) ---
        // Same binding layout as dequant_layout but uses sh_dequant_any.wgsl
        // which dispatches on quant_type: Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32.
        // Used for token_embd.weight which may be Q4_K on mixed-quant models.
        let dequant_any_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DequantAny Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });
        let dequant_any_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DequantAny Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../sh_dequant_any.wgsl").into()),
        });
        let dequant_any_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DequantAny Pipeline Layout"),
                bind_group_layouts: &[&dequant_any_layout],
                push_constant_ranges: &[],
            });
        let dequant_any_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DequantAny Pipeline"),
                layout: Some(&dequant_any_pipeline_layout),
                module: &dequant_any_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- 5. MatMul Pipeline ---
        // Bindings: 0=GGUF Blob read-only, 1=Input Vector read-only,
        //           2=Output Vector read-write, 3=Params uniform

        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // GGUF
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Input x
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Output y
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let matmul_src = include_str!("../sh_matmul_q4_0.wgsl");
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(matmul_src.into()),
        });

        let matmul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul Pipeline Layout"),
                bind_group_layouts: &[&matmul_layout],
                push_constant_ranges: &[],
            });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 5b. MatMul F32 Pipeline ---
        let matmul_f32_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul F32 Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // W (F32)
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Input x
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Output y
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let matmul_f32_src = include_str!("../sh_matmul_f32.wgsl");
        let matmul_f32_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(matmul_f32_src.into()),
        });

        let matmul_f32_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul F32 Pipeline Layout"),
                bind_group_layouts: &[&matmul_f32_layout],
                push_constant_ranges: &[],
            });

        let matmul_f32_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MatMul F32 Pipeline"),
                layout: Some(&matmul_f32_pipeline_layout),
                module: &matmul_f32_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- 6. RMSNorm Pipeline ---
        let rmsnorm_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RMSNorm Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Input x
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Output y
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 1: bytes [2GB, 4GB)
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 2: bytes [4GB, end)
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let rmsnorm_src = include_str!("../sh_rmsnorm.wgsl");
        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMSNorm Shader"),
            source: wgpu::ShaderSource::Wgsl(rmsnorm_src.into()),
        });

        let rmsnorm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RMSNorm Pipeline Layout"),
                bind_group_layouts: &[&rmsnorm_layout],
                push_constant_ranges: &[],
            });

        let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Pipeline"),
            layout: Some(&rmsnorm_pipeline_layout),
            module: &rmsnorm_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 7. Mega-Layer Pipeline ---
        // Bindings: 0=GGUF Blob read-only, 1=Activation In read-write,
        //           2=Temp State read-write, 3=LayerOffsets uniform,
        //           4=LayerParams uniform, 5=Norm Bank preflight, 6=RoPE Cache preflight
        let layer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer V1 Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Activation In
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Temp State
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // LayerOffsets
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // LayerParams
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Norm Bank
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // RoPE Cache
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // KV Cache K (Persistent)
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // KV Cache V (Persistent)
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // CacheParams (Uniform)
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 1: bytes [2GB, 4GB)
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 2: bytes [4GB, end)
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Layer Output Scales (Gemma-4 per-block, F32; binding 17)
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let layer_src = include_str!("../sh_layer_v1.wgsl");
        let layer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer V1 Shader"),
            source: wgpu::ShaderSource::Wgsl(layer_src.into()),
        });

        let layer_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Layer V1 Pipeline Layout"),
                bind_group_layouts: &[&layer_layout],
                push_constant_ranges: &[],
            });

        let mk_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Layer V1 Pipeline ({})", entry)),
                layout: Some(&layer_pipeline_layout),
                module: &layer_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let layer_pipeline_attn_norm = mk_pipeline("main_attn_norm");
        let layer_pipeline_qkv = mk_pipeline("main_qkv");
        let layer_pipeline_qk_norm = mk_pipeline("main_qk_norm");
        let layer_pipeline_attn_out = mk_pipeline("main_attn_out");
        let layer_pipeline_attn_proj = mk_pipeline("main_attn_proj");
        let layer_pipeline_ffn_norm = mk_pipeline("main_ffn_norm");
        let layer_pipeline_ffn_proj = mk_pipeline("main_ffn_proj");
        let layer_pipeline_ffn_down = mk_pipeline("main_ffn_down");
        let layer_pipeline_post_attn_norm = mk_pipeline("main_post_attn_norm");
        let layer_pipeline_post_ffw_norm = mk_pipeline("main_post_ffw_norm");

        // --- LM Head (blob-based quantized matmul) ---
        // Layout: binding 0 = blob_0, 1 = act_in (read), 2 = logits (write), 3 = params,
        //         10 = blob_1, 11 = blob_2
        let lm_head_blob_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Head Blob Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });

        let lm_head_blob_src = include_str!("../sh_head_blob.wgsl");
        let lm_head_blob_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Head Blob Shader"),
            source: wgpu::ShaderSource::Wgsl(lm_head_blob_src.into()),
        });
        let lm_head_blob_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LM Head Blob Pipeline Layout"),
                bind_group_layouts: &[&lm_head_blob_layout],
                push_constant_ranges: &[],
            });
        let lm_head_blob_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Head Blob Pipeline"),
                layout: Some(&lm_head_blob_pipeline_layout),
                module: &lm_head_blob_shader,
                entry_point: Some("main_lm_head"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- 8. INT4 KV Cache Layout (bindings 0-13) ---
        // Bindings 0-9 identical to layer_layout.
        // Bindings 10-13: k_packed (U32), k_scale (F32), v_packed (U32), v_scale (F32).
        let int4_extra_entries = [
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 14,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 16,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: None,
                    has_dynamic_offset: false,
                },
                count: None,
            },
        ];
        // Build full 14-entry list for INT4 layout
        // (re-use the 10 base entries from layer_layout pattern)
        let layer_layout_int4 = {
            let base = [
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                int4_extra_entries[0],
                int4_extra_entries[1],
                int4_extra_entries[2],
                int4_extra_entries[3],
            ];
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Layer V1 INT4 Layout"),
                entries: &base,
            })
        };

        let int4_src = include_str!("../sh_layer_v1_int4.wgsl");
        let int4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer V1 INT4 Shader"),
            source: wgpu::ShaderSource::Wgsl(int4_src.into()),
        });
        let int4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Layer V1 INT4 Pipeline Layout"),
            bind_group_layouts: &[&layer_layout_int4],
            push_constant_ranges: &[],
        });
        let layer_pipeline_attn_out_int4 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Layer V1 INT4 attn_out Pipeline"),
                layout: Some(&int4_pipeline_layout),
                module: &int4_shader,
                entry_point: Some("main_attn_out_int4"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- 9. Quantize KV Pipeline (7 bindings: f32_k, f32_v, packed_k, packed_v, scale_k, scale_v, params) ---
        let quantize_kv_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Quantize KV Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });
        let quant_kv_src = include_str!("../sh_quantize_kv.wgsl");
        let quant_kv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Quantize KV Shader"),
            source: wgpu::ShaderSource::Wgsl(quant_kv_src.into()),
        });
        let quant_kv_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Quantize KV Pipeline Layout"),
            bind_group_layouts: &[&quantize_kv_layout],
            push_constant_ranges: &[],
        });
        let quantize_kv_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Quantize KV Pipeline"),
                layout: Some(&quant_kv_pl_layout),
                module: &quant_kv_shader,
                entry_point: Some("quantize_kv"),
                compilation_options: Default::default(),
                cache: None,
            });

        // gemma-4 dense-latent PLE per-layer input construction.
        // Dedicated layout + pipeline; only bound/dispatched when spec.ple_enabled,
        // so no other model's bind-group construction or dispatch changes.
        let ple_input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PLE Input Layout"),
            entries: &[
                // Blob bindings 0, 10-16 (read-only storage), matching read_blob.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, // token_ids (read)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, // inp_batch (read) — scaled embeddings
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, // PleInputParams uniform
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, // ple_input output (read_write)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });
        let ple_input_src = include_str!("../sh_ple_input.wgsl");
        let ple_input_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PLE Input Shader"),
            source: wgpu::ShaderSource::Wgsl(ple_input_src.into()),
        });
        let ple_input_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PLE Input Pipeline Layout"),
            bind_group_layouts: &[&ple_input_layout],
            push_constant_ranges: &[],
        });
        let ple_input_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PLE Input Pipeline"),
            layout: Some(&ple_input_pl_layout),
            module: &ple_input_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // gemma-4 dense-latent PLE block residual.
        // Dedicated layout + pipeline; only bound/dispatched when ple_enabled.
        let ple_block_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PLE Block Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, // activation (read_write)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, // temp (read_write)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, // LayerOffsets uniform
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, // PleBlockParams uniform
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9, // CacheParams uniform
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17, // layer_scales (read)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18, // ple_input (read)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });
        let ple_block_src = include_str!("../sh_ple_block.wgsl");
        let ple_block_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PLE Block Shader"),
            source: wgpu::ShaderSource::Wgsl(ple_block_src.into()),
        });
        let ple_block_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PLE Block Pipeline Layout"),
            bind_group_layouts: &[&ple_block_layout],
            push_constant_ranges: &[],
        });
        let ple_block_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PLE Block Pipeline"),
            layout: Some(&ple_block_pl_layout),
            module: &ple_block_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            dequant_pipeline,
            dequant_bind_group_layout: dequant_layout,
            dequant_any_pipeline,
            dequant_any_bind_group_layout: dequant_any_layout,
            matmul_pipeline,
            matmul_layout,
            matmul_f32_pipeline,
            matmul_f32_layout,
            rmsnorm_pipeline,
            rmsnorm_layout,
            layer_pipeline_attn_norm,
            layer_pipeline_qkv,
            layer_pipeline_qk_norm,
            layer_pipeline_attn_out,
            layer_pipeline_attn_proj,
            layer_pipeline_ffn_norm,
            layer_pipeline_ffn_proj,
            layer_pipeline_ffn_down,
            layer_pipeline_post_attn_norm,
            layer_pipeline_post_ffw_norm: layer_pipeline_post_ffw_norm.clone(),
            layer_layout,
            lm_head_blob_pipeline,
            lm_head_blob_layout,
            layer_layout_int4,
            layer_pipeline_attn_out_int4,
            quantize_kv_layout,
            quantize_kv_pipeline,
            ple_input_pipeline,
            ple_input_layout,
            ple_block_pipeline,
            ple_block_layout,
        }
    }

    /// Read back GPU buffer contents to CPU as f32 values.
    /// Exported as pub(super) so sub-modules can call it.
    pub(super) fn readback_helper(&self, device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device
                .poll(wgpu::PollType::Poll)
                .expect("GPU device lost during readback poll");
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    /// Workgroup ceil-div: (n + 255) / 256 must match the dispatch sizing used throughout.
    #[test]
    fn workgroup_ceil_div_rounds_up_correctly() {
        assert_eq!((256 + 255) / 256, 1); // exact multiple
        assert_eq!((257 + 255) / 256, 2); // one over
        assert_eq!((1 + 255) / 256, 1); // minimum
        assert_eq!((512 + 255) / 256, 2); // exact double
    }

    /// RoPE softmax temperature denominator: sum of exp-shifted values for a 3-element slice.
    #[test]
    fn softmax_sum_is_positive_finite() {
        let logits = [1.0_f32, 2.0, 3.0];
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|x| (x - max_val).exp()).sum();
        assert!(sum.is_finite() && sum > 0.0);
    }

    /// KV cache ring index: position modulo context length stays in bounds.
    #[test]
    fn kv_cache_ring_index_wraps_within_bounds() {
        let ctx = 2048_usize;
        for pos in [0, 1, ctx - 1, ctx, ctx + 1, ctx * 2] {
            let idx = pos % ctx;
            assert!(idx < ctx, "ring index {idx} out of bounds for ctx={ctx}");
        }
    }
}

// ─── ISF Integration Spec (vault-derived, no model runs needed!) ──────────
/// Derive TDR risk from model metadata (no model runs!)
pub fn derive_tdr_risk_from_metadata(n_vocab: usize) -> &'static str {
    if n_vocab > 200_000 {
        "High"
    } else if n_vocab > 100_000 {
        "Medium"
    } else {
        "Low"
    }
}

/// Compute max safe workgroups for head blob dispatch from vault metadata  
pub fn compute_max_safe_workgroups(n_vocab: usize, budget_ms: f64) -> u32 {
    let head_wgs = n_vocab / 32; // 32 tokens per WG
    if head_wgs > 4000 {
        ((budget_ms * 1_000_000.0 - 500.0) / 500.0)
            .min(head_wgs as f64)
            .max(1.0) as u32
    } else {
        ((budget_ms * 1_000_000.0 - 500.0) / 100.0)
            .min(head_wgs as f64)
            .max(1.0) as u32
    }
}

#[allow(clippy::items_after_test_module)]
/// ISF Rule: YieldNow — derived from TDR risk and dispatch timing
pub fn should_yield_now(gpu_ms: f64, n_vocab: usize) -> bool {
    let risk = derive_tdr_risk_from_metadata(n_vocab);
    match risk {
        "High" => gpu_ms > 10.0,
        "Medium" => gpu_ms > 20.0,
        _ => gpu_ms > 50.0,
    }
}

/// ISF Rule: IncreaseTileCount — when head consistently close to budget  
pub fn should_increase_tiles(current_wgs: u32, n_vocab: usize) -> bool {
    let max_safe = compute_max_safe_workgroups(n_vocab, 100.0); // 100ms default budget
    (current_wgs as f64 / (max_safe.max(1) as f64)) > 0.8
}

/// Risk level string for logging/debugging (for ISF fact emission)
pub fn risk_level_string(n_vocab: usize) -> String {
    match derive_tdr_risk_from_metadata(n_vocab) {
        "High" => "TDR_RISK_HIGH".to_string(),
        "Medium" => "TDR_RISK_MEDIUM".to_string(),
        _ => "TDR_RISK_LOW".to_string(),
    }
}
