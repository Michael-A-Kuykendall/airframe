//! invariant_probe — PPT Invariant Cage capture probe.
//!
//! Loads a GGUF model on the GPU, runs the golden fixture `[BOS, Hello]` through
//! the production forward path (`GpuRuntime::generate_isf`), and emits the
//! per-layer `LayerOutput` + `FinalLogits` activations as JSON on stdout.
//!
//! This is the CAPTURE side of the PPT invariant cage. The CERTIFY side
//! (`tests/test_invariants.rs`) runs this binary per model and compares the
//! emitted RMS/checksum against the golden `vault/vault.duckdb` `layer_oracles`.
//!
//! Usage:
//!   invariant_probe <gguf_path> <model_name>
//!
//! Output (stdout, single JSON line):
//!   {"model": "...", "layers":[{layer_idx,position,rms,checksum}],
//!    "final_logits":{position,rms,checksum}}
//!
//! All stderr is diagnostic only. The capture is GATED: it only fires when
//! `AIRFRAME_CAPTURE_INVARIANT=1` AND a session is registered — so this binary
//! has zero effect on normal inference.

use airframe::backend::bindless::pipeline::inference::{
    clear_invariant_capture_sink, set_invariant_capture_sink,
};
use airframe::runtime::gpu::GpuRuntime;
use airframe_observe::facts::CapturedLayer;
use serde::Serialize;
use shimmytok::Tokenizer;
use std::path::Path;

#[derive(Serialize)]
struct LayerJson {
    layer_idx: u32,
    position: u32,
    rms: f32,
    checksum: i64,
}

#[derive(Serialize)]
struct FinalJson {
    position: u32,
    rms: f32,
    checksum: i64,
}

#[derive(Serialize)]
struct EmbedDiag {
    quant_type: u32,
    weight_offset: u64,
    row_bytes: u64,
    bos_token_id: u32,
    hello_token_id: u32,
    bos_first_20: Vec<f32>,
    hello_first_20: Vec<f32>,
}

#[derive(Serialize)]
struct ProbeOutput {
    model: String,
    layers: Vec<LayerJson>,
    final_logits: Option<FinalJson>,
    embed_diag: EmbedDiag,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: invariant_probe <gguf_path> <model_name>");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let model_name = &args[2];

    // Golden fixture: prompt "Hello" tokenizes to [BOS, Hello] with add_special=true,
    // matching the vault oracle position=1 rows.
    let prompt = "Hello";

    // Enable capture + install the sink that layer outputs are appended into.
    std::env::set_var("AIRFRAME_CAPTURE_INVARIANT", "1");
    let mut sink: Vec<CapturedLayer> = Vec::new();
    set_invariant_capture_sink(&mut sink);

    let rt = GpuRuntime::load(Path::new(model_path)).await?;

    let params = airframe::runtime::gpu::SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 1,
        seed: 0,
        extra_stop_tokens: vec![],
    };

    // Runs the golden forward; forward_fn appends FinalLogits, the layer loop
    // appends a LayerOutput per layer into `sink`.
    let _text = rt.generate_isf(prompt, &params, None)?;

    clear_invariant_capture_sink();

    // ── Embedding diagnostics ──────────────────────────────────────────
    let tokens = rt
        .tokenizer()
        .encode("Hello", true)
        .unwrap_or_default();
    let bos_id = tokens.first().copied().unwrap_or(1);
    let hello_id = tokens.get(1).copied().unwrap_or(15043);
    let bos_embd = rt.dequant_token_embd(bos_id);
    let hello_embd = rt.dequant_token_embd(hello_id);
    // RMS and checksum of the raw Hello embedding (input to layer 0)
    let hello_rms = (hello_embd.iter().map(|x| x*x).sum::<f32>() / hello_embd.len() as f32).sqrt();
    use airframe_observe::facts::{rms, checksum as cs};
    eprintln!("[diag] Hello embedding: rms={:.6} checksum={}", hello_rms, cs(&hello_embd));

    // ── Verify attn_norm weight tensor directly ──────────────────────────
    // Dequant the attn_norm weight and print first 5 values.
    let nw = rt.dequant_tensor_f32("blk.0.attn_norm.weight", 8);
    eprintln!("[diag] blk.0.attn_norm.first8={:?}", nw.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", "));
    // Print the exact absolute offset from Rust metadata
    let ann_off = rt.tensor_offset("blk.0.attn_norm.weight").unwrap_or(0);
    eprintln!("[diag] blk.0.attn_norm.weight abs_offset={}", ann_off);
    let fw = rt.dequant_tensor_f32("blk.0.ffn_norm.weight", 8);
    eprintln!("[diag] blk.0.ffn_norm.first8={:?}", fw.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", "));
    let ow = rt.dequant_tensor_f32("output_norm.weight", 8);
    eprintln!("[diag] output_norm.first8={:?}", ow.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", "));

    // ── Norm weight type diagnostic ────────────────────────────────────
    // Check what quant type the norm weights are stored as.
    // The preflight code copies dim*4 bytes as F32 — if they're F16, that's a buffer overread.
    let spec = rt.spec();
    eprintln!("[diag] rms_eps={} temp_buffer_size={} dim={} ff_dim={}",
        spec.rms_eps, spec.temp_buffer_size, spec.n_embd, spec.ff_dim);
    eprintln!("[diag] n_head={} n_head_kv={} head_dim={} n_layer={}",
        spec.n_head, spec.n_head_kv, spec.head_dim, spec.n_layer);
    eprintln!("[diag] embd_quant_type={} row_bytes={} embd_weight_offset={}",
        rt.embd_quant_type(), rt.row_bytes(), rt.embd_weight_offset());

    let embed_diag = EmbedDiag {
        quant_type: rt.embd_quant_type(),
        weight_offset: rt.embd_weight_offset(),
        row_bytes: rt.row_bytes(),
        bos_token_id: bos_id,
        hello_token_id: hello_id,
        bos_first_20: bos_embd.into_iter().take(20).collect(),
        hello_first_20: hello_embd.into_iter().take(20).collect(),
    };

    let mut layers = Vec::new();
    let mut final_logits = None;
    for c in &sink {
        if c.is_final_logits {
            final_logits = Some(FinalJson {
                position: c.position,
                rms: c.rms,
                checksum: c.checksum,
            });
        } else {
            layers.push(LayerJson {
                layer_idx: c.layer_idx,
                position: c.position,
                rms: c.rms,
                checksum: c.checksum,
            });
        }
    }
    layers.sort_by_key(|l| (l.position, l.layer_idx));

    let out = ProbeOutput {
        model: model_name.clone(),
        layers,
        final_logits,
        embed_diag,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
