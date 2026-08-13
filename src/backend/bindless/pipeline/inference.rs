//! Full-model inference dispatch methods for `BindlessPipeline`.
// TODO: break run_full_model_prefill_chunked_with_cache_state into a separate chunking helper once prefill chunking is the default path.
use super::super::loader::BindlessModel;
use super::*;
use crate::backend::tdr::TdrScheduler;
use crate::backend::tdr_calibration;
use crate::core::routing::ModelRoutePlan;
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;
/// Result type for model inference returning three activation vectors
type InferenceResult = Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>;

// ── PPT Invariant Capture Hook ───────────────────────────────────────────────
// Gated behind the `isf` feature (which also gates `airframe_observe`). When the
// invariant probe sets `AIRFRAME_CAPTURE_INVARIANT=1`, each transformer layer's
// post-layer activation is read back from the GPU and appended to a global
// in-memory sink. This is the CAPTURE side of the PPT invariant cage — it must
// NOT run in normal inference (the gate short-circuits before any GPU work).
#[cfg(feature = "isf")]
pub static mut INVARIANT_CAPTURE: Option<*mut Vec<airframe_observe::facts::CapturedLayer>> = None;

/// A single captured layer activation (rms + checksum, lightweight — no full
/// vector retained, to keep the sink small across many layers/positions).
#[cfg(feature = "isf")]
#[derive(Clone)]
pub struct CapturedLayer {
    pub layer_idx: u32,
    pub position: u32,
    pub rms: f32,
    pub checksum: i64,
    pub is_final_logits: bool,
}

/// Install the capture sink. `sink` must outlive the forward pass.
#[cfg(feature = "isf")]
pub fn set_invariant_capture_sink(sink: &mut Vec<airframe_observe::facts::CapturedLayer>) {
    // SAFETY: the probe owns `sink` for the duration of the forward pass; we
    // store the pointer and only deref it synchronously inside the forward call.
    unsafe {
        INVARIANT_CAPTURE = Some(sink as *mut Vec<airframe_observe::facts::CapturedLayer>);
    }
}

/// Clear the capture sink (call after the forward pass completes).
#[cfg(feature = "isf")]
pub fn clear_invariant_capture_sink() {
    unsafe {
        INVARIANT_CAPTURE = None;
    }
}

/// Borrow the registered capture sink mutably, if one is set.
#[cfg(feature = "isf")]
pub fn invariant_capture_sink_mut(
) -> Option<&'static mut Vec<airframe_observe::facts::CapturedLayer>> {
    unsafe { INVARIANT_CAPTURE.map(|p| &mut *p) }
}

// ── Layer-dump capture (q1c req 4: fused multi-token prefill states) ─────────
// Env-gated full-state capture for layer_dump_gpu: after each transformer layer
// of a fused multi-token prefill, the LAST token's activation slice is read
// back and pushed into a global sink drained by the dump tool. Unlike the
// isf-gated invariant capture (rms+checksum only), the full hidden state is
// retained so the dump tool can emit per-layer stats and run strict
// layer-to-layer divergence checks (q1c Gate 1). NOT isf-gated — plain release
// builds must work. No-op unless AIRFRAME_LAYER_DUMP_CAPTURE=1.
#[derive(Clone, Debug)]
pub struct LayerDumpState {
    pub layer_idx: u32,
    pub position: u32,
    pub hidden_states: Vec<f32>,
}

static LAYER_DUMP_STATES: OnceLock<Mutex<Vec<LayerDumpState>>> = OnceLock::new();

pub fn layer_dump_capture_enabled() -> bool {
    std::env::var("AIRFRAME_LAYER_DUMP_CAPTURE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Drain (and clear) all captured layer states.
pub fn layer_dump_drain() -> Vec<LayerDumpState> {
    let sink = LAYER_DUMP_STATES.get_or_init(|| Mutex::new(Vec::new()));
    std::mem::take(&mut *sink.lock().unwrap())
}

fn emit_layer_dump_state(
    layer_idx: u32,
    position: u32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    activation_buffer: &wgpu::Buffer,
    offset: u64,
    byte_len: u64,
) {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("Layer Dump Staging {}", layer_idx)),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("Layer Dump Capture {}", layer_idx)),
    });
    enc.copy_buffer_to_buffer(activation_buffer, offset, &staging, 0, byte_len);
    queue.submit(Some(enc.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU device lost during layer dump capture");
    let slice = staging.slice(..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .expect("GPU device lost during layer dump poll");
        if let Ok(res) = rx.try_recv() {
            res.expect("layer dump capture buffer map failed");
            break;
        }
    }
    let mapped = slice.get_mapped_range();
    let vals: &[f32] = bytemuck::cast_slice(&mapped);
    LAYER_DUMP_STATES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(LayerDumpState {
            layer_idx,
            position,
            hidden_states: vals.to_vec(),
        });
    drop(mapped);
    staging.unmap();
}
// ── Stack dump capture (OBS-1 / PEEL): residual + per-stage intercepts ────────
// Enabled when a stack sink is installed (stack_dump_gpu / AIRFRAME_STACK_DUMP).
// Shares the same GPU readback as invariant capture when both are active.
// Stage snaps follow airframe/docs/PEEL_STRUCTURE.md (correct buffer+offset+count).
#[cfg(feature = "isf")]
#[derive(Clone, Debug)]
pub struct StageSnap {
    pub name: String,
    pub rms: f32,
    pub first8: Vec<f32>,
    pub nan_count: u32,
    pub count: u32,
    pub buffer: String,
    pub offset_elems: u32,
    pub sampled: String,
}

#[cfg(feature = "isf")]
#[derive(Clone, Debug)]
pub struct StackLayerSnap {
    pub layer_idx: u32,
    pub position: u32,
    pub rms: f32,
    pub first8: Vec<f32>,
    pub nan_count: u32,
    pub residual_in: Option<StageSnap>,
    pub stages: Vec<StageSnap>,
}

#[cfg(feature = "isf")]
pub static mut STACK_LAYER_CAPTURE: Option<*mut Vec<StackLayerSnap>> = None;

#[cfg(feature = "isf")]
pub fn set_stack_layer_capture_sink(sink: &mut Vec<StackLayerSnap>) {
    unsafe {
        STACK_LAYER_CAPTURE = Some(sink as *mut Vec<StackLayerSnap>);
    }
}

#[cfg(feature = "isf")]
pub fn clear_stack_layer_capture_sink() {
    unsafe {
        STACK_LAYER_CAPTURE = None;
    }
}

#[cfg(feature = "isf")]
pub fn stack_layer_capture_sink_mut() -> Option<&'static mut Vec<StackLayerSnap>> {
    unsafe { STACK_LAYER_CAPTURE.map(|p| &mut *p) }
}

// ── PPT Invariant Per-Tensor Capture Hook ───────────────────────────────────
// Mirrors the layer sink but carries the per-kernel activation stats
// (q/k/v/post-attn/ffn/output RMS+checksum) for one transformer layer. This is
// what lets the certify loop PIN a broken sub-kernel (QKV vs Attn vs FFN) once
// a layer diverges. Populated by `run_layer_with_cache_debug`, which already
// reads those tensors back into CPU vecs — so this is zero extra GPU work.
#[cfg(feature = "isf")]
pub static mut INVARIANT_PTENSOR_CAPTURE: Option<*mut Vec<CapturedPerTensor>> = None;

/// One layer's per-kernel activation stats (lightweight — RMS+checksum only).
#[cfg(feature = "isf")]
#[derive(Clone)]
pub struct CapturedPerTensor {
    pub layer_idx: u32,
    pub position: u32,
    pub q_rms: f32,
    pub q_checksum: i64,
    pub k_rms: f32,
    pub k_checksum: i64,
    pub v_rms: f32,
    pub v_checksum: i64,
    pub post_rms: f32,
    pub post_checksum: i64,
    pub ffn_rms: f32,
    pub ffn_checksum: i64,
    pub output_rms: f32,
    pub output_checksum: i64,
}

#[cfg(feature = "isf")]
pub fn set_invariant_ptensor_capture_sink(sink: &mut Vec<CapturedPerTensor>) {
    unsafe {
        INVARIANT_PTENSOR_CAPTURE = Some(sink as *mut Vec<CapturedPerTensor>);
    }
}

#[cfg(feature = "isf")]
pub fn clear_invariant_ptensor_capture_sink() {
    unsafe {
        INVARIANT_PTENSOR_CAPTURE = None;
    }
}

#[cfg(feature = "isf")]
pub fn invariant_ptensor_capture_sink_mut() -> Option<&'static mut Vec<CapturedPerTensor>> {
    unsafe { INVARIANT_PTENSOR_CAPTURE.map(|p| &mut *p) }
}

/// Factory bead P1: mid-layer stage TRACE.
/// Enabled when `AIRFRAME_TRACE_STAGES=1`, or when multi-token prefill runs with
/// `AIRFRAME_TRACE_PREFILL_LAYERS=1` (so certify_family packages get stages).
fn stage_trace_enabled(batch_size: u32) -> bool {
    if std::env::var("AIRFRAME_TRACE_STAGES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return true;
    }
    batch_size > 1
        && std::env::var("AIRFRAME_TRACE_PREFILL_LAYERS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

/// Force-yield pending GPU work, copy `n_f32` floats from `src`@`offset` into
/// `readback`, return stats. `readback` must be ≥ n_f32 * 4 bytes.
#[allow(clippy::too_many_arguments)]
fn peel_stage_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tdr: &mut TdrScheduler<'_>,
    src: &wgpu::Buffer,
    offset: u64,
    n_f32: u32,
    readback: &wgpu::Buffer,
    layer: usize,
    stage: &str,
    buffer_name: &str,
    offset_elems: u32,
    first_nan_stage: &mut Option<String>,
    log_trace: bool,
) -> Result<StageSnapHost, String> {
    let n_f32 = n_f32.max(1);
    let byte_len = (n_f32 as u64) * 4;
    let label = format!("peel-L{}-{}", layer, stage);
    tdr.force_yield(&label)?;

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&label),
    });
    enc.copy_buffer_to_buffer(src, offset, readback, 0, byte_len);
    queue.submit(Some(enc.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| format!("GPU lost during stage peel ({})", label))?;
    tdr.reset_accumulator();

    let slice = readback.slice(..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|_| format!("GPU lost during stage peel poll ({})", label))?;
        if let Ok(res) = rx.try_recv() {
            res.map_err(|_| format!("stage peel map failed ({})", label))?;
            break;
        }
    }
    let mapped = slice.get_mapped_range();
    let vals: &[f32] = bytemuck::cast_slice(&mapped);
    let nan_count = vals.iter().filter(|&&x| x.is_nan()).count() as u32;
    let first8: Vec<f32> = vals.iter().take(8).copied().collect();
    let sum_sq: f32 = vals.iter().map(|x| x * x).sum();
    let rms = if vals.is_empty() {
        0.0
    } else {
        (sum_sq / vals.len() as f32).sqrt()
    };
    if log_trace {
        let first5: Vec<f32> = vals.iter().take(5).copied().collect();
        eprintln!(
            "[STAGE-TRACE] layer={} stage={} nan={}/{} rms={:.6} first5={:?} buf={} off={} count={}",
            layer,
            stage,
            nan_count,
            vals.len(),
            rms,
            first5,
            buffer_name,
            offset_elems,
            n_f32
        );
    }
    if nan_count > 0 && first_nan_stage.is_none() {
        let tag = format!("layer{}/{}", layer, stage);
        eprintln!("[STAGE-TRACE] FIRST_NAN_STAGE={}", tag);
        *first_nan_stage = Some(tag);
    }
    let snap = StageSnapHost {
        name: stage.to_string(),
        rms,
        first8,
        nan_count,
        count: n_f32,
        buffer: buffer_name.to_string(),
        offset_elems,
        sampled: "real".to_string(),
    };
    drop(mapped);
    readback.unmap();
    Ok(snap)
}

/// Host-side stage snap (always available; converted to `StageSnap` under `isf`).
#[derive(Clone, Debug)]
struct StageSnapHost {
    name: String,
    rms: f32,
    first8: Vec<f32>,
    nan_count: u32,
    count: u32,
    buffer: String,
    offset_elems: u32,
    sampled: String,
}

#[cfg(feature = "isf")]
impl StageSnapHost {
    fn into_stage_snap(self) -> StageSnap {
        StageSnap {
            name: self.name,
            rms: self.rms,
            first8: self.first8,
            nan_count: self.nan_count,
            count: self.count,
            buffer: self.buffer,
            offset_elems: self.offset_elems,
            sampled: self.sampled,
        }
    }
}

/// Read back the post-layer activation and append a `CapturedLayer` to the sink.
/// Mirrors the existing `trace_prefill_layers` readback (copy → submit → poll →
/// map → read → unmap) but routes the values into the probe's capture sink.
/// `stages` / `residual_in` are PEEL intercepts already read during the layer loop.
#[cfg(feature = "isf")]
#[allow(clippy::too_many_arguments)]
fn emit_layer_capture(
    layer_idx: u32,
    position: u32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    activation_buffer: &wgpu::Buffer,
    readback_buffer: &wgpu::Buffer,
    offset: u64,
    byte_len: u64,
    residual_in: Option<StageSnap>,
    stages: Vec<StageSnap>,
) {
    let inv_on = std::env::var("AIRFRAME_CAPTURE_INVARIANT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let inv_sink = if inv_on {
        invariant_capture_sink_mut()
    } else {
        None
    };
    let stack_sink = stack_layer_capture_sink_mut();
    if inv_sink.is_none() && stack_sink.is_none() {
        return;
    }
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("Invariant Capture Layer {}", layer_idx)),
    });
    enc.copy_buffer_to_buffer(activation_buffer, offset, readback_buffer, 0, byte_len);
    queue.submit(Some(enc.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU device lost during invariant capture");
    let slice = readback_buffer.slice(..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    loop {
        device
            .poll(wgpu::PollType::Poll)
            .expect("GPU device lost during invariant capture poll");
        if let Ok(res) = rx.try_recv() {
            res.expect("invariant capture buffer map failed");
            break;
        }
    }
    let mapped = slice.get_mapped_range();
    let vals: &[f32] = bytemuck::cast_slice(&mapped);
    let rms = airframe_observe::facts::rms(vals);
    let checksum = airframe_observe::facts::checksum(vals);
    let nan_count = vals.iter().filter(|x| x.is_nan()).count() as u32;
    let first8: Vec<f32> = vals.iter().take(8).copied().collect();
    if let Some(sink) = inv_sink {
        sink.push(airframe_observe::facts::CapturedLayer {
            layer_idx,
            position,
            rms,
            checksum,
            is_final_logits: false,
        });
    }
    if let Some(sink) = stack_sink {
        sink.push(StackLayerSnap {
            layer_idx,
            position,
            rms,
            first8,
            nan_count,
            residual_in,
            stages,
        });
    }
    drop(mapped);
    readback_buffer.unmap();
}

/// Push a layer's per-kernel activation stats into the per-tensor capture sink.
/// `q/k/v/post/ffn/output` are the CPU-readback vecs from `run_layer_with_cache_debug`.
/// Gated identically to `emit_layer_capture` (env + sink) so it is a no-op in
/// normal inference and when no per-tensor sink is installed.
#[cfg(feature = "isf")]
#[allow(clippy::too_many_arguments)]
pub fn emit_ptensor_capture(
    layer_idx: u32,
    position: u32,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    post: &[f32],
    ffn: &[f32],
    output: &[f32],
) {
    if !std::env::var("AIRFRAME_CAPTURE_INVARIANT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return;
    }
    let sink = match invariant_ptensor_capture_sink_mut() {
        Some(s) => s,
        None => return,
    };
    sink.push(CapturedPerTensor {
        layer_idx,
        position,
        q_rms: airframe_observe::facts::rms(q),
        q_checksum: airframe_observe::facts::checksum(q),
        k_rms: airframe_observe::facts::rms(k),
        k_checksum: airframe_observe::facts::checksum(k),
        v_rms: airframe_observe::facts::rms(v),
        v_checksum: airframe_observe::facts::checksum(v),
        post_rms: airframe_observe::facts::rms(post),
        post_checksum: airframe_observe::facts::checksum(post),
        ffn_rms: airframe_observe::facts::rms(ffn),
        ffn_checksum: airframe_observe::facts::checksum(ffn),
        output_rms: airframe_observe::facts::rms(output),
        output_checksum: airframe_observe::facts::checksum(output),
    });
}

impl BindlessPipeline {
    pub fn run_full_model(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        spec: &ModelSpec,
    ) -> Vec<f32> {
        self.run_full_model_with_cache(
            device,
            queue,
            model,
            input_embd,
            head_weights_override,
            0,
            1,
            spec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_full_model_with_cache(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        seq_len: u32,
        spec: &ModelSpec,
    ) -> Vec<f32> {
        self.run_full_model_with_cache_state(
            device,
            queue,
            model,
            input_embd,
            head_weights_override,
            current_pos,
            seq_len,
            None,
            spec,
        )
        .expect("GPU forward pass failed")
        .2
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_full_model_prefill_chunked_with_cache_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
        spec: &ModelSpec,
        chunk_tokens: u32,
    ) -> InferenceResult {
        let dim = spec.n_embd;
        assert!(dim > 0, "spec.n_embd must be > 0");
        assert!(
            input_embd.len().is_multiple_of(dim),
            "input_embd must align to token rows"
        );
        assert!(chunk_tokens > 0, "chunk_tokens must be > 0");

        // Gemma-4 input embedding scale: llama.cpp gemma4.cpp applies
        // `ggml_scale(inpL, sqrtf(n_embd))` to token embeddings. RMSNorm makes
        // it invisible to normed stages, but pre-norm projections (PLE context
        // branch, MoE routers, decode logits) read inpL directly.
        let scaled_embd: std::borrow::Cow<'_, [f32]>;
        let embd_slice: &[f32] = if spec.scale_embeddings_by_sqrt_dim {
            let s = (dim as f32).sqrt();
            scaled_embd = std::borrow::Cow::Owned(input_embd.iter().map(|v| v * s).collect());
            scaled_embd.as_ref()
        } else {
            input_embd
        };

        let trace_chunks = std::env::var("AIRFRAME_TRACE_PREFILL_CHUNKS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let total_tokens = embd_slice.len() / dim;
        if total_tokens == 0 {
            return self.run_full_model_with_cache_state(
                device,
                queue,
                model,
                embd_slice,
                head_weights_override,
                current_pos,
                current_pos,
                kv_state,
                spec,
            );
            // ^ Return type is now Result, so this propagates Ok or Err correctly.
        }

        let chunk_rows = chunk_tokens as usize;
        let mut processed_tokens = 0u32;
        let mut last_result = None;

        for (chunk_idx, chunk) in embd_slice.chunks(chunk_rows * dim).enumerate() {
            let chunk_token_count = (chunk.len() / dim) as u32;
            let chunk_current_pos = current_pos + processed_tokens;
            let chunk_seq_len = chunk_current_pos + chunk_token_count;

            if trace_chunks {
                eprintln!(
                    "[PREFILL] chunk={} tokens={} current_pos={} seq_len={}",
                    chunk_idx, chunk_token_count, chunk_current_pos, chunk_seq_len
                );
            }

            last_result = Some(self.run_full_model_with_cache_state(
                device,
                queue,
                model,
                chunk,
                head_weights_override,
                chunk_current_pos,
                chunk_seq_len,
                kv_state,
                spec,
            )?);

            if trace_chunks {
                eprintln!("[PREFILL] chunk={} complete", chunk_idx);
            }

            processed_tokens += chunk_token_count;
        }

        last_result.ok_or_else(|| "chunked prefill produced no chunks".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_full_model_with_cache_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        seq_len: u32,
        kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
        spec: &ModelSpec,
    ) -> InferenceResult {
        // Derive all constants from ModelSpec
        let dim = spec.n_embd as u32;
        let layer_count = spec.n_layer;
        let vocab_size = spec.n_vocab as u32;
        let ffn_dim = spec.ff_dim as u32;
        let temp_stride = spec.temp_buffer_size as u32;

        // Phase 4a escape hatch: set AIRFRAME_PINGPONG_ACTIVATION=1 to enable ping-pong.
        // Default off until Steps 3-4 are verified.
        let use_pingpong = std::env::var("AIRFRAME_PINGPONG_ACTIVATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let weight_quant_type = model
            .metadata
            .get_tensor_type("blk.0.attn_q.weight")
            .unwrap_or(2);
        let qt_v = model
            .metadata
            .get_tensor_type("blk.0.attn_v.weight")
            .unwrap_or(weight_quant_type);
        let qt_ffn_down = model
            .metadata
            .get_tensor_type("blk.0.ffn_down.weight")
            .unwrap_or(weight_quant_type);
        let packed_quant_type = weight_quant_type | (qt_v << 8) | (qt_ffn_down << 16);
        let _ = packed_quant_type; // per-layer quant is computed in the loop below

        // 1. Buffers
        let batch_size = (input_embd.len() as u32) / dim;
        // A. Activation (Residual Stream) - Init with Embeddings
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation A"),
            contents: bytemuck::cast_slice(input_embd),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // A2. Activation B (Ping-Pong partner).
        // Only created when ping-pong is active to avoid wasting VRAM on the old path.
        // When use_pingpong=false, activation_buffer_b is a dummy zero-byte buffer
        // that is never actually bound or used.
        let activation_buffer_b = if use_pingpong {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Activation B (Ping-Pong)"),
                contents: bytemuck::cast_slice(input_embd), // same initial residual
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        } else {
            // Dummy 1-byte buffer — never bound, just satisfies the type system.
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Activation B (disabled)"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        // B. Temp Buffer
        // Needs to hold FFN Gate + Up + scratch space per token
        let temp_buffer_size = batch_size as u64 * temp_stride as u64 * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // C. Layer Params (computed per-layer below; placeholder base for struct copy)
        // NOTE: quant_type varies per layer in mixed-quant models (e.g. Q4_K_M).
        //       Per-layer params buffers are created inside the layer loop.
        let use_route_v2_layer_params = std::env::var("SHIMMY_ROUTE_V2_LAYER_PARAMS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let route_plan = use_route_v2_layer_params.then(|| {
            ModelRoutePlan::from_spec_and_tensors(spec, |name| {
                model.metadata.tensor_offsets.contains_key(name)
            })
        });
        let ffn_kind_policy = route_plan
            .as_ref()
            .map(ModelRoutePlan::ffn_kind_policy_code)
            .unwrap_or(ModelRoutePlan::FFN_KIND_INFER);
        let qkv_layout_policy = route_plan
            .as_ref()
            .map(ModelRoutePlan::qkv_layout_policy_code)
            .unwrap_or(ModelRoutePlan::QKV_LAYOUT_INFER);

        let params_base = LayerParams {
            dim,
            head_count: spec.n_head as u32,
            head_count_kv: spec.n_head_kv as u32,
            head_dim: spec.head_dim as u32,
            rope_dim: spec.rope_dim as u32,
            rms_eps: spec.rms_eps,
            ffn_dim,
            temp_stride,
            quant_qk: 0,
            quant_v: 0,
            quant_attn_out: 0,
            quant_ffn_down: 0,
            quant_ffn_gate: 0,
            quant_ffn_up: 0,
            attn_logit_softcap: spec.attn_logit_softcap,
            post_norm_enabled: spec.post_norm_enabled as u32,
            qk_norm_enabled: spec.has_qk_norm as u32,
            layer_norm_enabled: spec.uses_layer_norm() as u32,
            ffn_kind_policy,
            qkv_layout_policy,
            batch_offset: 0,
            batch_count: batch_size,
            q_weight_k: 0,
            k_weight_k: 0,
            formula_qk: 0,
            formula_v: 0,
            formula_attn_out: 0,
            formula_ffn_down: 0,
            formula_ffn_gate: 0,
            formula_ffn_up: 0,
            blob_base_words: 0,
            chunk_words: 0,
            v_plain_rms_norm: spec.v_plain_rms_norm as u32,
            out_scale_enabled: spec.out_scale_enabled as u32,
            ple_latent_dim: spec.ple_latent_dim as u32,
            ple_enabled: spec.ple_enabled as u32,
            attn_scale_override: 0.0,
        };

        // Adaptive QKV micro-batch chunk size.
        // Reads SHIMMY_PREFILL_CHUNK; defaults to 1 (safest — one token per dispatch).
        // Users with fast GPUs can raise this; Q4_K_M on RTX 3060 is safe at 1.
        // A future TIMESTAMP_QUERY calibration pass will auto-tune this at model load.
        let qkv_chunk: u32 = std::env::var("SHIMMY_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .clamp(1, batch_size.max(1));

        // ── TDR Scheduler ────────────────────────────────────────────────────
        // TdrScheduler owns the command encoder and tracks accumulated GPU time.
        // It replaces the scattered tdr_submit_poll! / tdr_yield_if_needed! macros
        // with clean, testable methods. Platform-aware budget (1400ms Windows,
        // 30000ms Linux/macOS). Override with SHIMMY_TDR_BUDGET_MS.
        //
        // Patent Notice: FSE + D0 Saturation Fabric scheduling.
        // Pending patent by Michael A. Kuykendall. All rights reserved.
        let mut tdr = TdrScheduler::new(device, queue, "Full Model", None);
        let tdr_log = std::env::var("AIRFRAME_LOG_TDR_POLLS")
            .map(|v| v == "1")
            .unwrap_or(false);

        // D. Output Logits
        // Only computed for the LAST token in the sequence (usually).
        // If we want all logits, we'd need batch_size * vocab_size.
        // For now, let's stick to last token logic for compatibility.
        let logits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Logits"),
            size: (vocab_size as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let l21_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L21 Final Norm Output"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pre_norm_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pre-Final-Norm Output"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // PEEL stage readback: must fit largest intercept (Q, attn_ctx, 2*ffn).
        let dim_q = (spec.n_head * spec.head_dim) as u32;
        let dim_kv = (spec.n_head_kv * spec.head_dim) as u32;
        let peel_max_elems = dim
            .max(dim_q)
            .max(dim_kv)
            .max(ffn_dim.saturating_mul(2))
            .max(1);
        let stage_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PEEL Stage Readback"),
            size: (peel_max_elems as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // E. KV Cache
        // Shader cache indexing is layer-local: [pos, kv_head, head_dim] with no layer axis.
        // Therefore full-model loop must bind a distinct K/V buffer per layer.
        // COPY_SRC required for PEEL K/V intercepts.
        let kv_size_per_buffer = spec.kv_cache_size_per_layer as u64;
        let local_kv_storage_per_layer = if kv_state.is_none() {
            let mut bufs = Vec::with_capacity(layer_count);
            for i in 0..layer_count {
                let kv_buffer_k = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV Cache K L{}", i)),
                    size: kv_size_per_buffer,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                let kv_buffer_v = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV Cache V L{}", i)),
                    size: kv_size_per_buffer,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                bufs.push((kv_buffer_k, kv_buffer_v));
            }
            Some(bufs)
        } else {
            None
        };

        // F. Cache Params
        let cache_params = CacheParams {
            current_pos,
            seq_len, // Total cached positions (including this batch)
            max_seq_len: spec.n_ctx as u32,
            batch_size,
            logical_pos_base: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        use crate::backend::bindless::loader::BlobWindow;

        // 2. Prepare Layers (Offsets & BindGroups)
        // For ping-pong: two bind group arrays — one with activation_buffer (A) at binding 1,
        // one with activation_buffer_b (B). Layer i uses set_a when i%2==0, set_b when i%2==1.
        // For the old path (use_pingpong=false): only set_a is used; set_b is empty.
        let mut layer_bind_groups = Vec::new(); // set A: activation_buffer at binding 1
        let mut layer_bind_groups_b = Vec::new(); // set B: activation_buffer_b at binding 1
        let mut _offset_buffers = Vec::new(); // Keep alive
        let mut _params_buffers: Vec<wgpu::Buffer> = Vec::new(); // Keep alive
        let mut _layer_params: Vec<LayerParams> = Vec::new(); // Per-layer params for QKV chunking
                                                              // Window info for each layer (for multi-resident-chunk models)
        let mut _layer_windows: Vec<Option<BlobWindow>> = Vec::new();
        let mut _layer_blob_base_words: Vec<u32> = Vec::new();

        for i in 0..layer_count {
            let compiled = &model.metadata.compiled_layers[i];
            let supported = [0u8, 1, 2, 6, 8, 12, 13, 14];
            for &q in &[
                compiled.quant_qk,
                compiled.quant_v,
                compiled.quant_attn_out,
                compiled.quant_ffn_down,
                compiled.quant_ffn_gate,
                compiled.quant_ffn_up,
            ] {
                if q != 0 && !supported.contains(&(q as u8)) {
                    panic!("Unsupported quant type {} in layer {}", q, i);
                }
            }
            let mut layer_params_i = LayerParams {
                quant_qk: compiled.quant_qk,
                quant_v: compiled.quant_v,
                quant_attn_out: compiled.quant_attn_out,
                quant_ffn_down: compiled.quant_ffn_down,
                quant_ffn_gate: compiled.quant_ffn_gate,
                quant_ffn_up: compiled.quant_ffn_up,
                formula_qk: formula_index_for_ggml(compiled.quant_qk),
                formula_v: formula_index_for_ggml(compiled.quant_v),
                formula_attn_out: formula_index_for_ggml(compiled.quant_attn_out),
                formula_ffn_down: formula_index_for_ggml(compiled.quant_ffn_down),
                formula_ffn_gate: formula_index_for_ggml(compiled.quant_ffn_gate),
                formula_ffn_up: formula_index_for_ggml(compiled.quant_ffn_up),
                blob_base_words: compiled.blob_base_words,
                chunk_words: model.chunk_words(),
                ..params_base
            };
            if spec.q_weight_k > 0 {
                layer_params_i.q_weight_k = spec.q_weight_k as u32;
            }
            if spec.k_weight_k > 0 {
                layer_params_i.k_weight_k = spec.k_weight_k as u32;
            }

            let (kv_buffer_k_ref, kv_buffer_v_ref): (&wgpu::Buffer, &wgpu::Buffer) =
                if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                    (&kv_k_layers[i], &kv_v_layers[i])
                } else {
                    let (local_k, local_v) = &local_kv_storage_per_layer
                        .as_ref()
                        .expect("local KV storage missing")[i];
                    (local_k, local_v)
                };

            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Offsets", i)),
                contents: bytemuck::bytes_of(&compiled.offsets),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // --- Window abstraction for multi-resident-chunk models ---
            // Determine the word span of this layer's tensors and create a window
            // covering the resident chunks needed. Adjust blob_base_words to be window-local.
            let (layer_window, layer_blob_base_words) =
                resolve_layer_window(model, &compiled.offsets, compiled.blob_base_words, i);

            // Update layer_params_i with window-adjusted blob_base_words
            // Need to recreate params_buffer_i with the adjusted value
            let layer_params_i = LayerParams {
                blob_base_words: layer_blob_base_words,
                ..layer_params_i
            };
            let params_buffer_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Params", i)),
                contents: bytemuck::bytes_of(&layer_params_i),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Build bind group with a specific activation buffer at binding 1.
            // This closure lets us create both A and B sets without duplicating all entries.
            let make_bg = |act_buf: &wgpu::Buffer, label: &str| {
                // Create blob bindings fresh for each bind group (BindingResource is not Copy)
                let blob_bindings = blob_bindings_for(model, layer_window.as_ref());

                // Clone binding resources since BindingResource is not Copy but cheap to clone
                let b0 = blob_bindings[0].clone();
                let b1 = blob_bindings[1].clone();
                let b2 = blob_bindings[2].clone();
                let b3 = blob_bindings[3].clone();
                let b4 = blob_bindings[4].clone();
                let b5 = blob_bindings[5].clone();
                let b6 = blob_bindings[6].clone();
                let b7 = blob_bindings[7].clone();

                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.layer_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: b0,
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: act_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: temp_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: params_buffer_i.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .norm_bank_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .rope_cache_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: kv_buffer_k_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: kv_buffer_v_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: cache_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: b1,
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: b2,
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: b3,
                        },
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: b4,
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: b5,
                        },
                        wgpu::BindGroupEntry {
                            binding: 15,
                            resource: b6,
                        },
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: b7,
                        },
                        wgpu::BindGroupEntry {
                            binding: 17,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .layer_scales_buffer
                                .as_entire_binding(),
                        },
                    ],
                })
            };

            // Set A: activation_buffer (always built — used by old path + even layers in pingpong)
            let bg_a = make_bg(&activation_buffer, &format!("Layer {} BG-A", i));

            // Set B: activation_buffer_b (only built when pingpong is active)
            if use_pingpong {
                let bg_b = make_bg(&activation_buffer_b, &format!("Layer {} BG-B", i));
                layer_bind_groups_b.push(bg_b);
            }

            _offset_buffers.push(buf);
            _params_buffers.push(params_buffer_i);
            _layer_params.push(layer_params_i);
            _layer_windows.push(layer_window);
            _layer_blob_base_words.push(layer_blob_base_words);
            layer_bind_groups.push(bg_a);
        }

        // 2b. Pre-build QKV chunk bind groups (Step 5: eliminate QKV forced polls).
        //
        // The QKV micro-batch loop currently patches `params_buffer_i` via write_buffer
        // for each chunk's batch_offset/batch_count, requiring 2 forced yields per chunk.
        // Instead: pre-build one params buffer + bind group per (layer, chunk_offset).
        // The layer loop then just selects the right pre-built bind group — no write_buffer,
        // no forced yields for QKV.
        //
        // _qkv_chunk_params_buffers[layer][chunk_idx] = pre-built params buffer
        // _qkv_chunk_bind_groups[layer][chunk_idx]   = pre-built bind group for that chunk
        //
        // For decode (batch_size=1, qkv_chunk=1): 1 chunk per layer = N_layers entries.
        // For prefill (batch_size=N, qkv_chunk=512): ceil(N/512) chunks per layer.
        let n_qkv_chunks = batch_size.div_ceil(qkv_chunk) as usize;
        let mut _qkv_chunk_params_buffers: Vec<Vec<wgpu::Buffer>> = Vec::with_capacity(layer_count);
        let mut _qkv_chunk_bind_groups: Vec<Vec<wgpu::BindGroup>> = Vec::with_capacity(layer_count);

        for i in 0..layer_count {
            let compiled = &model.metadata.compiled_layers[i];
            let layer_params_base = _layer_params[i];
            // Use window-adjusted blob_base_words for multi-resident-chunk models
            let layer_blob_base_words = _layer_blob_base_words[i];
            let layer_window = &_layer_windows[i];
            let (kv_buffer_k_ref, kv_buffer_v_ref): (&wgpu::Buffer, &wgpu::Buffer) =
                if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                    (&kv_k_layers[i], &kv_v_layers[i])
                } else {
                    let (local_k, local_v) = &local_kv_storage_per_layer
                        .as_ref()
                        .expect("local KV storage missing")[i];
                    (local_k, local_v)
                };
            let offsets_buf = &_offset_buffers[i];

            let mut layer_chunk_params: Vec<wgpu::Buffer> = Vec::with_capacity(n_qkv_chunks);
            let mut layer_chunk_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(n_qkv_chunks);

            let mut qkv_offset: u32 = 0;
            while qkv_offset < batch_size {
                let this_chunk = (batch_size - qkv_offset).min(qkv_chunk);
                let chunk_params = LayerParams {
                    batch_offset: qkv_offset,
                    batch_count: this_chunk,
                    blob_base_words: layer_blob_base_words,
                    ..layer_params_base
                };
                let chunk_params_buf =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("L{}-QKV-chunk{}-Params", i, qkv_offset)),
                        contents: bytemuck::bytes_of(&chunk_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                // Create blob bindings for this chunk (window-aware)
                let blob_bindings = blob_bindings_for(model, layer_window.as_ref());
                // Clone since BindingResource is not Copy
                let b0 = blob_bindings[0].clone();
                let b1 = blob_bindings[1].clone();
                let b2 = blob_bindings[2].clone();
                let b3 = blob_bindings[3].clone();
                let b4 = blob_bindings[4].clone();
                let b5 = blob_bindings[5].clone();
                let b6 = blob_bindings[6].clone();
                let b7 = blob_bindings[7].clone();

                let chunk_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("L{}-QKV-chunk{}-BG", i, qkv_offset)),
                    layout: &self.layer_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: b0,
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: activation_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: temp_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: offsets_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: chunk_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .norm_bank_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .rope_cache_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: kv_buffer_k_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: kv_buffer_v_ref.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: cache_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: b1,
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: b2,
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: b3,
                        },
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: b4,
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: b5,
                        },
                        wgpu::BindGroupEntry {
                            binding: 15,
                            resource: b6,
                        },
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: b7,
                        },
                        wgpu::BindGroupEntry {
                            binding: 17,
                            resource: model
                                .preflight
                                .as_ref()
                                .unwrap()
                                .layer_scales_buffer
                                .as_entire_binding(),
                        },
                    ],
                });
                layer_chunk_params.push(chunk_params_buf);
                layer_chunk_bgs.push(chunk_bg);
                qkv_offset += this_chunk;
            }
            let _ = compiled; // used above in outer loop
            _qkv_chunk_params_buffers.push(layer_chunk_params);
            _qkv_chunk_bind_groups.push(layer_chunk_bgs);
        }

        // 3. Final Norm
        let norm_weight = model
            .metadata
            .get_tensor_offset("output_norm.weight")
            .expect("output_norm missing");
        let norm_bias = model
            .metadata
            .get_tensor_offset("output_norm.bias")
            .map(|off| (off / 4) as u32)
            .unwrap_or(0);
        // word index (byte_offset / 4); safe: 4.4GB/4 = 1.1B < u32::MAX
        let norm_weight_words = (norm_weight / 4) as u32;

        // Window for final norm (output_norm.weight + bias). sh_rmsnorm.wgsl has
        // no blob_base_words uniform — it indexes read_blob with weight_offset /
        // bias_offset directly — so those offsets must themselves be rebased to
        // the window start.
        let norm_window = model
            .rmsnorm_window(
                norm_weight_words,
                if norm_bias != 0 {
                    Some(norm_bias)
                } else {
                    None
                },
                dim,
            )
            .expect("final norm tensor span exceeds window capacity");
        let norm_window_base = norm_window.window_base_words();

        // 0 is the shader's "bias disabled" sentinel. If a real bias rebased to
        // exactly 0 it would be silently dropped, so refuse rather than emit
        // wrong math.
        let norm_bias_local = if norm_bias != 0 {
            let local = norm_bias - norm_window_base;
            assert_ne!(
                local, 0,
                "output_norm.bias sits exactly at the window base; rebased offset \
                 collides with the bias-disabled sentinel"
            );
            local
        } else {
            0
        };

        let norm_params = RMSNormParams {
            count: dim,
            weights_offset: norm_weight_words - norm_window_base,
            bias_offset: norm_bias_local,
            eps: spec.rms_eps,
            norm_type: if spec.uses_layer_norm() { 1 } else { 0 },
            chunk_words: model.chunk_words(),
        };
        let norm_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Final Norm Params"),
            contents: bytemuck::bytes_of(&norm_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Offset for the LAST token in the batch
        let last_token_offset = (batch_size as u64 - 1u64) * (dim as u64) * 4u64;
        let token_size = std::num::NonZeroU64::new((dim as u64) * 4u64).unwrap();

        // Create blob bindings for final norm (window-aware)
        let blob_bindings = norm_window.binding_resources(model);
        // Clone since BindingResource is not Copy
        let b0 = blob_bindings[0].clone();
        let b1 = blob_bindings[1].clone();
        let b2 = blob_bindings[2].clone();
        let b3 = blob_bindings[3].clone();
        let b4 = blob_bindings[4].clone();
        let b5 = blob_bindings[5].clone();
        let b6 = blob_bindings[6].clone();
        let b7 = blob_bindings[7].clone();

        let norm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Final Norm BG"),
            layout: &self.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b0,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &activation_buffer,
                        offset: last_token_offset,
                        size: Some(token_size),
                    }),
                },
                // Use temp_buffer for output to avoid read-write aliasing on activation_buffer
                // Output to the BEGINNING of temp_buffer (reusing space)
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &temp_buffer,
                        offset: 0,
                        size: Some(token_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: norm_param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: b1,
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: b2,
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: b3,
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: b4,
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: b5,
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: b6,
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: b7,
                },
            ],
        });

        // 4. Output Head
        // When head_weights_override = Some(buf): diagnostic F32 matmul override path.
        // When head_weights_override = None (default): blob-based quantized head — reads
        //   output.weight directly from the GGUF blob, no dequant buffer required.
        let head_tensor_name = if model.metadata.get_tensor_type("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let head_weight_off = (model
            .metadata
            .get_tensor_offset(head_tensor_name)
            .unwrap_or(0)
            / 4) as u32;
        let head_quant_type = model
            .metadata
            .get_tensor_type(head_tensor_name)
            .unwrap_or(2);
        let supported = [0u8, 1, 2, 6, 8, 12, 13, 14];
        if !supported.contains(&(head_quant_type as u8)) {
            panic!("Unsupported head quant type {}", head_quant_type);
        }

        enum HeadBg {
            F32(wgpu::BindGroup),
            Blob(wgpu::BindGroup),
        }

        let head_bg = if let Some(override_buf) = head_weights_override {
            // --- Diagnostic F32 override (kept for shimmy_eval comparison tests) ---
            let head_params = MatMulParams {
                n: vocab_size,
                k: dim,
                weights_offset: head_weight_off,
                padding: 0,
            };
            let head_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Head Params F32"),
                contents: bytemuck::bytes_of(&head_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            HeadBg::F32(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG F32"),
                layout: &self.matmul_f32_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: override_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &temp_buffer,
                            offset: 0,
                            size: Some(token_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: logits_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: head_param_buf.as_entire_binding(),
                    },
                ],
            }))
        } else {
            // --- Default blob-based path: output.weight stays quantized on GPU ---
            // Window for the LM head weight rows. sh_head_blob.wgsl indexes
            // read_blob as `weight_off + rel_word` with no separate base
            // uniform, so weight_off must be rebased to the window start.
            let head_window = model
                .lm_head_window(0, vocab_size, dim)
                .expect("LM head tensor span exceeds window capacity");

            let head_params = HeadBlobParams {
                vocab_size,
                dim,
                weight_off: rebase_head_weight_off(head_weight_off, &head_window),
                formula_index: formula_index_for_ggml(head_quant_type),
                softcap: spec.final_logit_softcap,
                base_row: 0,
                chunk_words: model.chunk_words(),
            };
            let head_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Head Params Blob"),
                contents: bytemuck::bytes_of(&head_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let blob_bindings = head_window.binding_resources(model);
            // Clone since BindingResource is not Copy
            let b0 = blob_bindings[0].clone();
            let b1 = blob_bindings[1].clone();
            let b2 = blob_bindings[2].clone();
            let b3 = blob_bindings[3].clone();
            let b4 = blob_bindings[4].clone();
            let b5 = blob_bindings[5].clone();
            let b6 = blob_bindings[6].clone();
            let b7 = blob_bindings[7].clone();

            HeadBg::Blob(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG Blob"),
                layout: &self.lm_head_blob_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b0,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &temp_buffer,
                            offset: 0,
                            size: Some(token_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: logits_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: head_param_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: b1,
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: b2,
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: b3,
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: b4,
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: b5,
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: b6,
                    },
                    wgpu::BindGroupEntry {
                        binding: 16,
                        resource: b7,
                    },
                ],
            }))
        };

        // 5. Command Encoding — managed by TdrScheduler (see tdr above).
        // The initial encoder was created by TdrScheduler::new().

        let wg_dim = dim.div_ceil(256);
        let ffn_total = ffn_dim * 2; // Gate + Up need this many threads
        let wg_ffn = ffn_total.div_ceil(256); // Ceil div by workgroup size (256)
        let wg_norm = dim.div_ceil(256);
        // matmul_f32 uses @workgroup_size(256).
        let wg_head_f32 = vocab_size.div_ceil(256);

        // QKV Dispatch Calculation
        let q_len = params_base.head_count * params_base.head_dim;
        let kv_len = params_base.head_count_kv * params_base.head_dim;
        let attn_dim = q_len; // n_head * head_dim (Qwen3: 4096 ≠ dim 2560)
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = total_qkv.div_ceil(256);
        let wg_qknorm = (q_len + kv_len).div_ceil(256); // must cover all Q+K elements, not just head_dim
                                                        // CRITICAL: attn_out writes [0..attn_dim), not [0..dim). Under-dispatch left
                                                        // elements dim..attn_dim-1 unwritten → garbage O-proj for Qwen3.
        let wg_attn = attn_dim.div_ceil(256);
        let trace_prefill_layers = std::env::var("AIRFRAME_TRACE_PREFILL_LAYERS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let disable_output_norm = std::env::var("SHIMMY_DISABLE_OUTPUT_NORM")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // P1: stage TRACE (layer 0 multi-token). Also on when prefill TRACE + batch>1.
        let do_stage_trace = stage_trace_enabled(batch_size);
        // PEEL: full stage intercepts when stack sink installed (all layers) or TRACE (L0).
        #[cfg(feature = "isf")]
        let stack_peel_all = stack_layer_capture_sink_mut().is_some();
        #[cfg(not(feature = "isf"))]
        let stack_peel_all = false;
        let do_peel = do_stage_trace || stack_peel_all;
        let mut first_nan_stage: Option<String> = None;
        let temp_last_offset = (batch_size as u64 - 1) * (temp_stride as u64) * 4;
        // Q lives at temp_base + dim; last-token Q byte offset
        let temp_q_offset = temp_last_offset + (dim as u64) * 4;
        // Last-token KV cache position (absolute)
        let last_kv_pos = current_pos + batch_size.saturating_sub(1);
        let kv_last_offset = (last_kv_pos as u64) * (kv_len as u64) * 4;
        let qk_norm_on = params_base.qk_norm_enabled != 0;

        if do_peel {
            eprintln!(
                "[PEEL] enabled batch_size={} current_pos={} seq_len={} dim={} dim_q={} dim_kv={} ffn_dim={} attn_dim={} stack_all={} trace={}",
                batch_size,
                current_pos,
                seq_len,
                dim,
                q_len,
                kv_len,
                ffn_dim,
                attn_dim,
                stack_peel_all,
                do_stage_trace
            );
            // Residual input (embeddings) for last token — before any layer
            let _ = peel_stage_readback(
                device,
                queue,
                &mut tdr,
                &activation_buffer,
                last_token_offset,
                dim,
                &stage_readback_buffer,
                0,
                "input",
                "activation",
                0,
                &mut first_nan_stage,
                do_stage_trace,
            )?;
        }

        // Loop Layers
        for (i, bg) in layer_bind_groups.iter().enumerate() {
            let params_layer = _layer_params[i]; // per-layer quant_type + base fields
            if trace_prefill_layers {
                eprintln!(
                    "[PREFILL-LAYER] start layer={} batch_size={} current_pos={} seq_len={}",
                    i, batch_size, current_pos, seq_len
                );
            }
            // TRACE logs layer 0; stack peel captures every layer when sink installed.
            let peel_this_layer = (do_stage_trace && i == 0) || stack_peel_all;
            let log_trace = do_stage_trace && i == 0;
            let mut residual_in_snap: Option<StageSnapHost> = None;
            let mut stages_host: Vec<StageSnapHost> = Vec::new();
            // All models use V1 pipelines. V1 handles all quant types (Q4_0, Q4_K, Q5_K,
            // Q6_K, F16, F32) via per-kernel quant_type branch checks and proven-correct
            // dequant helpers. The Q4K-specific shader family has been removed.
            let (
                pipe_attn_norm,
                pipe_qkv,
                pipe_qk_norm,
                pipe_attn_out,
                pipe_attn_proj,
                pipe_post_attn_norm,
                pipe_ffn_norm,
                pipe_ffn_proj,
                pipe_ffn_down,
                pipe_post_ffw_norm,
            ) = (
                &self.layer_pipeline_attn_norm,
                &self.layer_pipeline_qkv,
                &self.layer_pipeline_qk_norm,
                &self.layer_pipeline_attn_out,
                &self.layer_pipeline_attn_proj,
                &self.layer_pipeline_post_attn_norm,
                &self.layer_pipeline_ffn_norm,
                &self.layer_pipeline_ffn_proj,
                &self.layer_pipeline_ffn_down,
                &self.layer_pipeline_post_ffw_norm,
            );

            // L.S0 residual_in (activation before AttnNorm)
            if peel_this_layer {
                residual_in_snap = Some(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &activation_buffer,
                    last_token_offset,
                    dim,
                    &stage_readback_buffer,
                    i,
                    "residual_in",
                    "activation",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }

            {
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - AttnNorm", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_attn_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            if peel_this_layer {
                // L.S1: temp[temp_base .. +dim]
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_last_offset,
                    dim,
                    &stage_readback_buffer,
                    i,
                    "attn_norm",
                    "temp",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            // QKV: micro-batched to avoid TDR on large batch prefill.
            // Step 5: uses pre-built per-chunk bind groups — no write_buffer, no forced yields.
            // For correctness, yield at layer boundary handles TDR protection.
            // For large batches, also yield every TDR_QKV_YIELD_INTERVAL chunks to prevent
            // the GPU accumulating >2s of work inside the QKV dispatch loop.
            {
                let chunk_bgs = &_qkv_chunk_bind_groups[i];
                let mut qkv_offset: u32 = 0;
                let mut chunk_idx: usize = 0;
                // Yield every 16 chunks for large batches (prevents TDR inside QKV loop)
                const QKV_YIELD_INTERVAL: usize = 16;
                while qkv_offset < batch_size {
                    let this_chunk = (batch_size - qkv_offset).min(qkv_chunk);
                    {
                        let mut cpass =
                            tdr.encoder
                                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some(&format!(
                                        "Layer {} QKV [{}/{}]",
                                        i, qkv_offset, batch_size
                                    )),
                                    timestamp_writes: None,
                                });
                        cpass.set_bind_group(0, &chunk_bgs[chunk_idx], &[]);
                        cpass.set_pipeline(pipe_qkv);
                        cpass.dispatch_workgroups(wg_qkv, this_chunk, 1);
                    }
                    // Yield every QKV_YIELD_INTERVAL chunks for large batches to prevent TDR
                    if batch_size > 1
                        && chunk_idx > 0
                        && chunk_idx.is_multiple_of(QKV_YIELD_INTERVAL)
                    {
                        let label = format!("layer-{}-qkv-interval-{}", i, chunk_idx);
                        tdr.force_yield(&label)?;
                    }
                    qkv_offset += this_chunk;
                    chunk_idx += 1;
                }
            }
            if peel_this_layer {
                // L.S2a Q: temp[temp_base + dim .. + dim_q]
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_q_offset,
                    q_len,
                    &stage_readback_buffer,
                    i,
                    "q",
                    "temp",
                    dim,
                    &mut first_nan_stage,
                    log_trace,
                )?);
                // L.S2b/S2c K/V at last token position in per-layer KV buffers
                let (kv_k_ref, kv_v_ref): (&wgpu::Buffer, &wgpu::Buffer) =
                    if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                        (&kv_k_layers[i], &kv_v_layers[i])
                    } else {
                        let (local_k, local_v) = &local_kv_storage_per_layer
                            .as_ref()
                            .expect("local KV storage missing")[i];
                        (local_k, local_v)
                    };
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    kv_k_ref,
                    kv_last_offset,
                    kv_len,
                    &stage_readback_buffer,
                    i,
                    "k",
                    "kv_k",
                    last_kv_pos * kv_len,
                    &mut first_nan_stage,
                    log_trace,
                )?);
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    kv_v_ref,
                    kv_last_offset,
                    kv_len,
                    &stage_readback_buffer,
                    i,
                    "v",
                    "kv_v",
                    last_kv_pos * kv_len,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            {
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - QKNorm", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_qk_norm);
                cpass.dispatch_workgroups(wg_qknorm, batch_size, 1);
            }
            if peel_this_layer && qk_norm_on {
                // L.S3a/S3b after in-place QK-norm
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_q_offset,
                    q_len,
                    &stage_readback_buffer,
                    i,
                    "q_norm",
                    "temp",
                    dim,
                    &mut first_nan_stage,
                    log_trace,
                )?);
                let (kv_k_ref, _): (&wgpu::Buffer, &wgpu::Buffer) =
                    if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                        (&kv_k_layers[i], &kv_v_layers[i])
                    } else {
                        let (local_k, local_v) = &local_kv_storage_per_layer
                            .as_ref()
                            .expect("local KV storage missing")[i];
                        (local_k, local_v)
                    };
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    kv_k_ref,
                    kv_last_offset,
                    kv_len,
                    &stage_readback_buffer,
                    i,
                    "k_norm",
                    "kv_k",
                    last_kv_pos * kv_len,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            {
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - AttnOut", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_attn_out);
                cpass.dispatch_workgroups(wg_attn, batch_size, 1);
            }
            if peel_this_layer {
                // L.S4 attn_ctx: temp[0..attn_dim]
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_last_offset,
                    attn_dim,
                    &stage_readback_buffer,
                    i,
                    "attn_ctx",
                    "temp",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            // TDR: yield after attn_out only if accumulated budget exceeded.
            // For large batch prefill: force yield after attn_out to prevent TDR.
            {
                let label = format!("layer-{}-attn_out", i);
                if batch_size > 8 {
                    tdr.force_yield(&label)?;
                } else {
                    tdr.yield_if_needed(&label)?;
                }
            }
            {
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - AttnProj", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_attn_proj);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            if peel_this_layer {
                // L.S5b residual after attn residual-add
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &activation_buffer,
                    last_token_offset,
                    dim,
                    &stage_readback_buffer,
                    i,
                    "attn_residual",
                    "activation",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            {
                // Post-attention norm correction (Gemma-2 only; no-op for post_norm_enabled==0)
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - PostAttnNorm", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_post_attn_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            if params_layer.quant_ffn_down != 12u32 {
                // For Q4K, ffn_norm is inside the Q4K ffn_proj kernel; skip V1 to avoid double norm.
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - FFNNorm", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_ffn_norm);
                cpass.dispatch_workgroups(1, batch_size, 1);
            }
            if peel_this_layer {
                // L.S7 ffn_norm output in temp[0..dim] (V1 path)
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_last_offset,
                    dim,
                    &stage_readback_buffer,
                    i,
                    "ffn_norm",
                    "temp",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            {
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - FFNProj", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_ffn_proj);
                cpass.dispatch_workgroups(wg_ffn, batch_size, 1);
            }
            if peel_this_layer {
                // L.S8a/S8b gate + up
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_last_offset,
                    ffn_dim,
                    &stage_readback_buffer,
                    i,
                    "ffn_gate",
                    "temp",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &temp_buffer,
                    temp_last_offset + (ffn_dim as u64) * 4,
                    ffn_dim,
                    &stage_readback_buffer,
                    i,
                    "ffn_up",
                    "temp",
                    ffn_dim,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            {
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - FFNDown", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_ffn_down);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
            if peel_this_layer {
                // L.S9b residual out
                stages_host.push(peel_stage_readback(
                    device,
                    queue,
                    &mut tdr,
                    &activation_buffer,
                    last_token_offset,
                    dim,
                    &stage_readback_buffer,
                    i,
                    "ffn_residual",
                    "activation",
                    0,
                    &mut first_nan_stage,
                    log_trace,
                )?);
            }
            // TDR: yield after ffn_down only if accumulated budget exceeded.
            {
                let label = format!("layer-{}-ffn_down", i);
                tdr.yield_if_needed(&label)?;
            }
            {
                // Post-FFW norm correction (Gemma-2 only; no-op for post_norm_enabled==0)
                let mut cpass = tdr
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Loop Layer {} - PostFfwNorm", i)),
                        timestamp_writes: None,
                    });
                cpass.set_bind_group(0, bg, &[]);
                cpass.set_pipeline(pipe_post_ffw_norm);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }

            // TDR: conditional yield at layer boundary.
            // NOTE: this yield is required for correctness on DeepSeek Q4K (UAV barrier).
            // For large batch prefill, always force a yield every layer to prevent TDR.
            // For decode (batch_size=1), only yield if budget exceeded (usually never).
            {
                let label = format!("layer-{}-boundary", i);
                if batch_size > 1 {
                    // Prefill: force yield every layer — GPU work per layer is ~0.5s on RTX 3060
                    // with 32-layer Q4K_M × large batch. Without this, TDR fires.
                    tdr.force_yield(&label)?;
                } else {
                    // Decode (single token): conditional only — budget rarely exceeded at 0.03s/step
                    tdr.yield_if_needed(&label)?;
                }
            }

            // ── PPT Invariant Capture (gated; no-op unless env + session set) ──
            // Emits a LayerOutput fact for layer `i` at `current_pos+1`, because
            // `current_pos` is the *start* index of the batch and the captured
            // activation is the LAST token in the batch (the one we care about).
            // For the golden [BOS,Hello] prefill (current_pos=0) this is position 1.
            #[cfg(feature = "isf")]
            {
                let residual_in = residual_in_snap.map(|s| s.into_stage_snap());
                let stages: Vec<StageSnap> = stages_host
                    .into_iter()
                    .map(|s| s.into_stage_snap())
                    .collect();
                emit_layer_capture(
                    i as u32,
                    current_pos + batch_size.saturating_sub(1),
                    device,
                    queue,
                    &activation_buffer,
                    &pre_norm_buffer,
                    last_token_offset,
                    (dim as u64) * 4,
                    residual_in,
                    stages,
                );
            }

            if layer_dump_capture_enabled() {
                emit_layer_dump_state(
                    i as u32,
                    current_pos + batch_size.saturating_sub(1),
                    device,
                    queue,
                    &activation_buffer,
                    last_token_offset,
                    (dim as u64) * 4,
                );
            }

            if trace_prefill_layers {
                // Must flush pending layer work first. For batch_size==1 the
                // layer-boundary path only yield_if_needed — without this force,
                // TRACE copy sees the *pre*-layer residual (embedding) for every
                // layer → false "identity residual" (Q3 diagnostic phantom).
                tdr.force_yield(&format!("layer-{}-pre-trace", i))?;
                // Layer trace readback — uses its own encoder after TDR flush.
                let mut trace_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("Layer {} Trace Readback", i)),
                    });
                trace_encoder.copy_buffer_to_buffer(
                    &activation_buffer,
                    last_token_offset,
                    &pre_norm_buffer,
                    0,
                    (dim as u64) * 4,
                );
                queue.submit(Some(trace_encoder.finish()));
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|_| {
                        "GPU device lost or TDR timeout during layer trace readback".to_string()
                    })?;
                tdr.reset_accumulator(); // readback did its own submit+poll

                let trace_slice = pre_norm_buffer.slice(..);
                let (tx_trace, rx_trace) = std::sync::mpsc::channel();
                trace_slice.map_async(wgpu::MapMode::Read, move |res| tx_trace.send(res).unwrap());
                loop {
                    device
                        .poll(wgpu::PollType::Poll)
                        .map_err(|_| "GPU device lost during layer trace poll".to_string())?;
                    if let Ok(res) = rx_trace.try_recv() {
                        res.map_err(|_| "Layer trace buffer map failed".to_string())?;
                        break;
                    }
                }
                let mapped = trace_slice.get_mapped_range();
                let trace_vals: &[f32] = bytemuck::cast_slice(&mapped);
                let nan_count = trace_vals.iter().filter(|&&x| x.is_nan()).count();
                let first5: Vec<f32> = trace_vals.iter().take(5).copied().collect();
                eprintln!(
                    "[PREFILL-LAYER-TRACE] layer={} nan={}/{} first5={:?}",
                    i,
                    nan_count,
                    trace_vals.len(),
                    first5
                );
                drop(mapped);
                pre_norm_buffer.unmap();
            }

            if trace_prefill_layers {
                eprintln!("[PREFILL-LAYER] complete layer={}", i);
            }
        }

        if do_peel {
            match &first_nan_stage {
                Some(s) => eprintln!("[PEEL] SUMMARY FIRST_NAN_STAGE={}", s),
                None => eprintln!("[PEEL] SUMMARY FIRST_NAN_STAGE=none"),
            }
        }

        // Snapshot h20 (post-layer-loop, pre-final-norm)
        if tdr_log {
            eprintln!(
                "[TDR-STATS] batch_size={} layers={} total_yields={} forced_per_layer_min={}",
                batch_size,
                layer_count,
                tdr.yield_count,
                if layer_count > 0 {
                    tdr.yield_count / layer_count as u32
                } else {
                    0
                }
            );
        }
        tdr.encoder.copy_buffer_to_buffer(
            &activation_buffer,
            last_token_offset,
            &pre_norm_buffer,
            0,
            (dim as u64) * 4,
        );

        // Final Norm — separate pass so wgpu inserts a memory barrier before the
        // LM Head pass reads from temp_buffer (same region that norm writes).
        if disable_output_norm {
            tdr.encoder.copy_buffer_to_buffer(
                &activation_buffer,
                last_token_offset,
                &temp_buffer,
                0,
                (dim as u64) * 4u64,
            );
        } else {
            let mut cpass = tdr
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Final Norm"),
                    timestamp_writes: None,
                });
            cpass.set_bind_group(0, &norm_bg, &[]);
            cpass.set_pipeline(&self.rmsnorm_pipeline);
            cpass.dispatch_workgroups(wg_norm, 1, 1);
        }
        // LM Head
        let mut _tile_bgs: Vec<wgpu::BindGroup> = Vec::new();
        {
            let mut cpass = tdr
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Head"),
                    timestamp_writes: None,
                });
            match &head_bg {
                HeadBg::Blob(bg) => {
                    cpass.set_pipeline(&self.lm_head_blob_pipeline);

                    // TDR-safe tiled dispatch: split into tiles of max_safe_wgs
                    let max_safe_wgs =
                        tdr_calibration::ensure_calibrated("unknown", "head_blob", dim);
                    let tile_size = 64u32; // @workgroup_size in sh_head_blob.wgsl
                    let total_wgs = vocab_size.div_ceil(tile_size);

                    if total_wgs <= max_safe_wgs {
                        // Fast path: single dispatch
                        cpass.set_bind_group(0, bg, &[]);
                        cpass.dispatch_workgroups(total_wgs, 1, 1);
                    } else {
                        // Tiled path: split into max_safe_wgs chunks
                        let mut dispatched = 0u32;

                        while dispatched < total_wgs {
                            let this_tile = (total_wgs - dispatched).min(max_safe_wgs);
                            let base_row = dispatched * tile_size;
                            let tile_idx = dispatched / max_safe_wgs;

                            let tile_params = HeadBlobParams {
                                vocab_size,
                                dim,
                                weight_off: 0, // rebased below, once the tile window is known
                                formula_index: formula_index_for_ggml(head_quant_type),
                                softcap: spec.final_logit_softcap,
                                base_row,
                                chunk_words: model.chunk_words(),
                            };

                            // Window for this tile's weight rows. weight_off is
                            // rebased onto it because sh_head_blob.wgsl adds it
                            // to read_blob indices directly.
                            let tile_window = model
                                .lm_head_window(base_row, this_tile * tile_size, dim)
                                .expect("LM head tile tensor span exceeds window capacity");
                            let tile_params = HeadBlobParams {
                                weight_off: rebase_head_weight_off(head_weight_off, &tile_window),
                                ..tile_params
                            };
                            let param_buf =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(&format!("Head Params tile-{}", tile_idx)),
                                    contents: bytemuck::bytes_of(&tile_params),
                                    usage: wgpu::BufferUsages::UNIFORM,
                                });

                            let blob_bindings = tile_window.binding_resources(model);
                            let b0 = blob_bindings[0].clone();
                            let b1 = blob_bindings[1].clone();
                            let b2 = blob_bindings[2].clone();
                            let b3 = blob_bindings[3].clone();
                            let b4 = blob_bindings[4].clone();
                            let b5 = blob_bindings[5].clone();
                            let b6 = blob_bindings[6].clone();
                            let b7 = blob_bindings[7].clone();

                            let tile_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some(&format!("Head BG tile-{}", tile_idx)),
                                layout: &self.lm_head_blob_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: b0,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Buffer(
                                            wgpu::BufferBinding {
                                                buffer: &temp_buffer,
                                                offset: 0,
                                                size: Some(token_size),
                                            },
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: logits_buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: param_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 10,
                                        resource: b1,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 11,
                                        resource: b2,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 12,
                                        resource: b3,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 13,
                                        resource: b4,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 14,
                                        resource: b5,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 15,
                                        resource: b6,
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 16,
                                        resource: b7,
                                    },
                                ],
                            });

                            cpass.set_bind_group(0, &tile_bg, &[]);
                            cpass.dispatch_workgroups(this_tile, 1, 1);

                            _tile_bgs.push(tile_bg);
                            dispatched += this_tile;
                        }
                    }
                }
                HeadBg::F32(bg) => {
                    cpass.set_bind_group(0, bg, &[]);
                    cpass.set_pipeline(&self.matmul_f32_pipeline);
                    cpass.dispatch_workgroups(wg_head_f32, 1, 1);
                }
            }
        }

        // 6. Readback
        let output_size = (vocab_size * 4) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        tdr.encoder
            .copy_buffer_to_buffer(&temp_buffer, 0, &l21_buffer, 0, (dim as u64) * 4);
        tdr.encoder
            .copy_buffer_to_buffer(&logits_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(tdr.encoder.finish()));

        let pre_norm_slice = pre_norm_buffer.slice(..);
        let (tx_pre, rx_pre) = std::sync::mpsc::channel();
        pre_norm_slice.map_async(wgpu::MapMode::Read, move |res| tx_pre.send(res).unwrap());

        let l21_slice = l21_buffer.slice(..);
        let (tx_l21, rx_l21) = std::sync::mpsc::channel();
        l21_slice.map_async(wgpu::MapMode::Read, move |res| tx_l21.send(res).unwrap());

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        let mut pre_done = false;
        let mut l21_done = false;
        let mut main_done = false;

        loop {
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|_| "GPU device lost during readback poll".to_string())?;

            if !pre_done {
                if let Ok(res) = rx_pre.try_recv() {
                    res.map_err(|_| {
                        "Pre-norm buffer map failed. Device lost or TDR timeout.".to_string()
                    })?;
                    pre_done = true;
                }
            }
            if !l21_done {
                if let Ok(res) = rx_l21.try_recv() {
                    res.map_err(|_| {
                        "L21 buffer map failed. Device lost or TDR timeout.".to_string()
                    })?;
                    l21_done = true;
                }
            }
            if !main_done {
                if let Ok(res) = rx.try_recv() {
                    res.map_err(|_| "Buffer map failed. Device lost or TDR timeout.".to_string())?;
                    main_done = true;
                }
            }

            if pre_done && l21_done && main_done {
                break;
            }
        }

        let pre_norm_data = pre_norm_slice.get_mapped_range();
        let pre_norm_result: Vec<f32> = bytemuck::cast_slice(&pre_norm_data).to_vec();
        drop(pre_norm_data);
        pre_norm_buffer.unmap();

        let l21_data = l21_slice.get_mapped_range();
        let l21_result: Vec<f32> = bytemuck::cast_slice(&l21_data).to_vec();
        drop(l21_data);
        l21_buffer.unmap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok((pre_norm_result, l21_result, result))
    }
}
