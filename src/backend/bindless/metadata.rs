use super::pipeline::CompiledLayerEntry;
use crate::core::spec::{GgufValue, ModelSpec};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// Byte size of one output row for a GGML quant type, given the in-dim
/// (columns) of the matrix. Mirrors the dequant row-stride used everywhere
/// else (e.g. fused-QKV splitting). Used to split a merged gate+up FFN tensor.
pub fn quant_type_row_bytes(qt: u32, cols: u64) -> u64 {
    match qt {
        0 => cols * 4,
        1 => cols * 2,
        2 => (cols / 32) * 18,
        8 => (cols / 32) * 34,
        12 => (cols / 256) * 144,
        13 => (cols / 256) * 176,
        14 => (cols / 256) * 210,
        _ => (cols / 32) * 18,
    }
}

/// Extracted metadata from GGUF header to locate tensors in the blob.
#[derive(Debug)]
pub struct BindlessMetadata {
    pub version: u32,
    pub tensor_count: u64,
    /// Tensor Name -> Byte Offset in GGUF file
    pub tensor_offsets: HashMap<String, u64>,
    /// Tensor Name -> GGML Type (0=F32, 1=F16, 2=Q4_0, 12=Q4_K, 14=Q6_K, etc.)
    pub tensor_types: HashMap<String, u32>,
    /// Tensor Name -> Dimensions (shape as Vec<u64>)
    pub tensor_dims: HashMap<String, Vec<u64>>,
    /// Header/Meta/Alignment overhead size (Data starts at this offset)
    pub data_start_offset: u64,
    /// All GGUF metadata key-value pairs
    pub gguf_metadata: HashMap<String, GgufValue>,
    /// Pre-compiled per-layer lookup table (FSE: built once at load, zero-cost at inference time).
    pub compiled_layers: Vec<CompiledLayerEntry>,
}

fn manifest_value(value: &shimmytok::gguf::GGUFValue) -> GgufValue {
    use shimmytok::gguf::GGUFValue as V;
    match value {
        V::U8(x) => GgufValue::U8(*x),
        V::I8(x) => GgufValue::I8(*x),
        V::U16(x) => GgufValue::U16(*x),
        V::I16(x) => GgufValue::I16(*x),
        V::U32(x) => GgufValue::U32(*x),
        V::I32(x) => GgufValue::I32(*x),
        V::U64(x) => GgufValue::U64(*x),
        V::I64(x) => GgufValue::I64(*x),
        V::F32(x) => GgufValue::F32(*x),
        V::F64(x) => GgufValue::F64(*x),
        V::Bool(x) => GgufValue::Bool(*x),
        V::String(x) => GgufValue::String(x.clone()),
        V::Array(x) => GgufValue::ArrayLen(x.len()),
    }
}

impl BindlessMetadata {
    /// scan a GGUF reader and extract tensor offsets.
    pub fn new<R: Read + Seek>(reader: &mut R) -> Self {
        Self::try_new(reader).expect("invalid GGUF manifest")
    }
    pub fn try_new<R: Read + Seek>(reader: &mut R) -> Result<Self, shimmytok::Error> {
        let manifest = {
            use std::io::SeekFrom;
            let file_len = reader.seek(SeekFrom::End(0))?;
            reader.seek(SeekFrom::Start(0))?;
            shimmytok::manifest::read_manifest(reader, file_len, Default::default())?
        };
        Self::from_manifest(&manifest)
    }
    pub fn from_manifest(
        manifest: &shimmytok::manifest::ModelManifest,
    ) -> Result<Self, shimmytok::Error> {
        let version = manifest.header().version;
        let tensor_count = manifest.header().tensor_count;
        let data_start = manifest.data_start();
        let gguf_metadata: HashMap<_, _> = manifest
            .header()
            .values
            .iter()
            .map(|(k, v)| (k.clone(), manifest_value(v)))
            .collect();
        let absolute_offsets: HashMap<_, _> = manifest
            .tensors()
            .iter()
            .map(|t| (t.name().to_owned(), t.absolute_offset()))
            .collect();
        let tensor_types: HashMap<_, _> = manifest
            .tensors()
            .iter()
            .map(|t| (t.name().to_owned(), t.ggml_type()))
            .collect();
        let tensor_dims: HashMap<_, _> = manifest
            .tensors()
            .iter()
            .map(|t| (t.name().to_owned(), t.dimensions().to_vec()))
            .collect();
        // FSE compiled-layer table: single pass over layer indices at load time.
        // Eliminates per-token format!/HashMap overhead from the inference hot path.
        let mut compiled_layers = Vec::new();
        let is_phi_arch = matches!(
            gguf_metadata.get("general.architecture"),
            Some(GgufValue::String(v)) if v == "phi"
        );
        {
            let _p = |offsets: &HashMap<String, u64>, layer: usize, s: &str| -> u32 {
                super::pipeline::pack_blob_offset(
                    offsets
                        .get(&format!("blk.{}.{}", layer, s))
                        .copied()
                        .unwrap_or(0),
                )
            };
            let t = |types: &HashMap<String, u32>, layer: usize, s: &str| -> u32 {
                types
                    .get(&format!("blk.{}.{}", layer, s))
                    .copied()
                    .unwrap_or(2) // default Q4_0
            };

            let mut layer_idx = 0usize;
            while absolute_offsets.contains_key(&format!("blk.{}.attn_norm.weight", layer_idx)) {
                // Per-layer base byte: one word before the aligned minimum tensor offset.
                let min_offset = absolute_offsets
                    .keys()
                    .filter(|k| k.starts_with(&format!("blk.{}.", layer_idx)))
                    .filter_map(|k| absolute_offsets.get(k).copied())
                    .filter(|&o| o > 0)
                    .min()
                    .unwrap_or(0);
                // Packed offset zero is reserved as the "tensor missing" sentinel.
                // Keep the first present tensor one word above the layer base.
                let base_byte = (min_offset & !3u64).saturating_sub(4);
                let blob_base_words = (base_byte / 4) as u32;
                // Shadow outer _p with per-layer relative offsets
                let p = |offsets: &HashMap<String, u64>, layer: usize, s: &str| -> u32 {
                    let abs = offsets
                        .get(&format!("blk.{}.{}", layer, s))
                        .copied()
                        .unwrap_or(0);
                    if abs == 0 {
                        0
                    } else {
                        super::pipeline::relative_packed_offset(abs, base_byte)
                            .expect("abs underflow")
                    }
                };
                // Optional tensor lookup with per-layer base
                let opt = |offsets: &std::collections::HashMap<String, u64>,
                           li: usize,
                           suffix: &str|
                 -> u32 {
                    let abs = *offsets.get(&format!("blk.{}.{}", li, suffix)).unwrap_or(&0);
                    if abs == 0 {
                        0
                    } else {
                        super::pipeline::relative_packed_offset(abs, base_byte)
                            .expect("abs underflow")
                    }
                };
                // Fused QKV support: phi-2, StarCoder2, GPT-2 and similar models store Q+K+V
                // in a single weight matrix `attn_qkv.weight`. When separate attn_q/k/v tensors
                // are absent, split the fused offset into per-component byte ranges.
                let fused_qkv_key = format!("blk.{}.attn_qkv.weight", layer_idx);
                let has_separate_q =
                    absolute_offsets.contains_key(&format!("blk.{}.attn_q.weight", layer_idx));
                let has_fused_qkv = absolute_offsets.contains_key(&fused_qkv_key);

                let (attn_q_off, attn_k_off, attn_v_off, lqt_main, lqt_v) = if has_separate_q {
                    let lm = t(&tensor_types, layer_idx, "attn_q.weight");
                    let lv = t(&tensor_types, layer_idx, "attn_v.weight");
                    (
                        p(&absolute_offsets, layer_idx, "attn_q.weight"),
                        p(&absolute_offsets, layer_idx, "attn_k.weight"),
                        p(&absolute_offsets, layer_idx, "attn_v.weight"),
                        lm,
                        lv,
                    )
                } else if has_fused_qkv {
                    let fused_off = *absolute_offsets.get(&fused_qkv_key).unwrap();
                    let fused_type = *tensor_types.get(&fused_qkv_key).unwrap_or(&2u32);
                    // dim_in = input columns (= n_embd); total_out = Q+K+V output rows
                    let dim_in = tensor_dims
                        .get(&fused_qkv_key)
                        .and_then(|d| d.first())
                        .copied()
                        .unwrap_or(0);
                    let total_out = tensor_dims
                        .get(&fused_qkv_key)
                        .and_then(|d| d.get(1))
                        .copied()
                        .unwrap_or(0);
                    // dim_q = n_head * head_dim; read from attn_output.weight's input dim
                    let attn_out_key = format!("blk.{}.attn_output.weight", layer_idx);
                    let dim_q = tensor_dims
                        .get(&attn_out_key)
                        .and_then(|d| d.first())
                        .copied()
                        .unwrap_or(dim_in);
                    // dim_k = dim_v = (total_out - dim_q) / 2  (handles GQA)
                    let dim_k = total_out.saturating_sub(dim_q) / 2;
                    // Bytes per output row based on quant type
                    let bpr: u64 = match fused_type {
                        0 => dim_in * 4,
                        1 => dim_in * 2,
                        2 => (dim_in / 32) * 18,
                        8 => (dim_in / 32) * 34,
                        12 => (dim_in / 256) * 144,
                        13 => (dim_in / 256) * 176,
                        14 => (dim_in / 256) * 210,
                        _ => (dim_in / 32) * 18,
                    };
                    let q_off = super::pipeline::relative_packed_offset(fused_off, base_byte)
                        .expect("abs underflow");
                    let k_off =
                        super::pipeline::relative_packed_offset(fused_off + dim_q * bpr, base_byte)
                            .expect("abs underflow");
                    let v_off = super::pipeline::relative_packed_offset(
                        fused_off + (dim_q + dim_k) * bpr,
                        base_byte,
                    )
                    .expect("abs underflow");
                    println!(
                        "[Metadata] Layer {}: fused QKV type={} dim_in={} dim_q={} dim_k={} bpr={} K@{} V@{}",
                        layer_idx, fused_type, dim_in, dim_q, dim_k, bpr, k_off, v_off
                    );
                    (q_off, k_off, v_off, fused_type, fused_type)
                } else {
                    (0u32, 0u32, 0u32, 2u32, 2u32)
                };

                let sep_q_bias = opt(&absolute_offsets, layer_idx, "attn_q.bias");
                let sep_k_bias = opt(&absolute_offsets, layer_idx, "attn_k.bias");
                let sep_v_bias = opt(&absolute_offsets, layer_idx, "attn_v.bias");
                let fused_qkv_bias_key = format!("blk.{}.attn_qkv.bias", layer_idx);
                let (attn_q_bias_off, attn_k_bias_off, attn_v_bias_off) =
                    if sep_q_bias != 0 || sep_k_bias != 0 || sep_v_bias != 0 {
                        (sep_q_bias, sep_k_bias, sep_v_bias)
                    } else if has_fused_qkv {
                        if let Some(&fused_bias_off) = absolute_offsets.get(&fused_qkv_bias_key) {
                            // Bias layout mirrors fused QKV rows: [Q rows][K rows][V rows], each f32.
                            let fused_qkv_key = format!("blk.{}.attn_qkv.weight", layer_idx);
                            let dim_in = tensor_dims
                                .get(&fused_qkv_key)
                                .and_then(|d| d.first())
                                .copied()
                                .unwrap_or(0);
                            let total_out = tensor_dims
                                .get(&fused_qkv_key)
                                .and_then(|d| d.get(1))
                                .copied()
                                .unwrap_or(0);
                            let attn_out_key = format!("blk.{}.attn_output.weight", layer_idx);
                            let dim_q = tensor_dims
                                .get(&attn_out_key)
                                .and_then(|d| d.first())
                                .copied()
                                .unwrap_or(dim_in);
                            let dim_k = total_out.saturating_sub(dim_q) / 2;
                            let q_bias =
                                super::pipeline::relative_packed_offset(fused_bias_off, base_byte)
                                    .expect("abs underflow");
                            let k_bias = super::pipeline::relative_packed_offset(
                                fused_bias_off + dim_q * 4,
                                base_byte,
                            )
                            .expect("abs underflow");
                            let v_bias = super::pipeline::relative_packed_offset(
                                fused_bias_off + (dim_q + dim_k) * 4,
                                base_byte,
                            )
                            .expect("abs underflow");
                            println!(
                                "[Metadata] Layer {}: fused QKV bias split Q@{} K@{} V@{}",
                                layer_idx, q_bias, k_bias, v_bias
                            );
                            (q_bias, k_bias, v_bias)
                        } else {
                            (0u32, 0u32, 0u32)
                        }
                    } else {
                        (0u32, 0u32, 0u32)
                    };

                let attn_norm_off = p(&absolute_offsets, layer_idx, "attn_norm.weight");
                let mut ffn_norm_off = p(&absolute_offsets, layer_idx, "ffn_norm.weight");
                if is_phi_arch && ffn_norm_off == 0 {
                    // Phi-family checkpoints can ship a single per-block norm; reuse attn_norm.
                    ffn_norm_off = attn_norm_off;
                }

                let attn_norm_bias_off = opt(&absolute_offsets, layer_idx, "attn_norm.bias");
                let mut ffn_norm_bias_off = opt(&absolute_offsets, layer_idx, "ffn_norm.bias");
                if is_phi_arch && ffn_norm_bias_off == 0 {
                    ffn_norm_bias_off = attn_norm_bias_off;
                }

                // Cache ffn_down quant type so the LayerOffsets builder below can read it.
                // (lqt_v comes from the QKV/separate-V branch above; lqt_main comes from the same.)
                let lqt_down = t(&tensor_types, layer_idx, "ffn_down.weight");

                // ── Merged gate+up FFN (SWIGLU: phi-3/4) ──────────────────────
                // phi-3/4 store gate and up CONCATENATED in one tensor
                // `ffn_up.weight` with dims [n_embd, 2*ff_dim] (no separate
                // `ffn_gate.weight`). The shader reads gate rows 0..ff_dim and
                // up rows ff_dim..2*ff_dim from the SAME tensor base. Without
                // this split, ffn_gate resolves to 0 and the shader treats the
                // model as NON-gated (GELU) — the phi gibberish root cause.
                // Detection is pure shape math: ffn_up output dim == 2*ffn_down
                // output dim AND no separate ffn_gate tensor.
                let up_key = format!("blk.{}.ffn_up.weight", layer_idx);
                let down_key = format!("blk.{}.ffn_down.weight", layer_idx);
                let has_gate_tensor =
                    absolute_offsets.contains_key(&format!("blk.{}.ffn_gate.weight", layer_idx));
                let ffn_up_abs = absolute_offsets.get(&up_key).copied().unwrap_or(0);
                // GGUF dims are [in, out]. ffn_up = [n_embd, 2*ff_dim];
                // ffn_down = [ff_dim, n_embd]. Merged iff up_out == 2 * down_in.
                let up_out_dim = tensor_dims
                    .get(&up_key)
                    .and_then(|d| d.get(1))
                    .copied()
                    .unwrap_or(0);
                let down_in_dim = tensor_dims
                    .get(&down_key)
                    .and_then(|d| d.first())
                    .copied()
                    .unwrap_or(0);
                let merged_gate_up = !has_gate_tensor
                    && ffn_up_abs != 0
                    && down_in_dim != 0
                    && up_out_dim == down_in_dim * 2;
                let (ffn_gate_off, ffn_up_off) = if merged_gate_up {
                    // Same tensor; gate = rows 0..ff_dim, up = rows ff_dim..2*ff_dim.
                    // Row byte size depends on the tensor's quant type.
                    let ff_dim_rows = down_in_dim;
                    let in_dim = tensor_dims
                        .get(&up_key)
                        .and_then(|d| d.first())
                        .copied()
                        .unwrap_or(0);
                    let bpr =
                        quant_type_row_bytes(t(&tensor_types, layer_idx, "ffn_up.weight"), in_dim);
                    let gate_rel = super::pipeline::relative_packed_offset(ffn_up_abs, base_byte)
                        .expect("abs underflow");
                    let up_rel = super::pipeline::relative_packed_offset(
                        ffn_up_abs + ff_dim_rows * bpr,
                        base_byte,
                    )
                    .expect("abs underflow");
                    println!(
                        "[Metadata] Layer {}: merged gate+up ffn_up dims={:?} -> gate@{} up@{} (ff_dim={}, bpr={})",
                        layer_idx,
                        tensor_dims.get(&up_key),
                        gate_rel,
                        up_rel,
                        ff_dim_rows,
                        bpr
                    );
                    (gate_rel, up_rel)
                } else {
                    (
                        p(&absolute_offsets, layer_idx, "ffn_gate.weight"),
                        p(&absolute_offsets, layer_idx, "ffn_up.weight"),
                    )
                };

                let offsets = super::pipeline::LayerOffsets {
                    attn_norm: attn_norm_off,
                    attn_norm_bias: attn_norm_bias_off,
                    attn_q: attn_q_off,
                    attn_k: attn_k_off,
                    attn_v: attn_v_off,
                    attn_out: p(&absolute_offsets, layer_idx, "attn_output.weight"),
                    ffn_norm: ffn_norm_off,
                    ffn_norm_bias: ffn_norm_bias_off,
                    ffn_gate: ffn_gate_off,
                    ffn_down: p(&absolute_offsets, layer_idx, "ffn_down.weight"),
                    ffn_up: ffn_up_off,
                    layer_idx: layer_idx as u32,
                    v_is_q4k: (lqt_v == 12) as u32,
                    ffn_down_is_q4k: (lqt_down == 12) as u32,
                    attn_q_norm: opt(&absolute_offsets, layer_idx, "attn_q_norm.weight"),
                    attn_k_norm: opt(&absolute_offsets, layer_idx, "attn_k_norm.weight"),
                    attn_q_bias: attn_q_bias_off,
                    attn_k_bias: attn_k_bias_off,
                    attn_v_bias: attn_v_bias_off,
                };
                let lqt_attn_out = t(&tensor_types, layer_idx, "attn_output.weight");
                let lqt_up = t(&tensor_types, layer_idx, "ffn_up.weight");
                // Non-gated FFN (StarCoder2 etc): ffn_gate.weight absent; use ffn_up's quant type
                // since the shader reads ffn_up weights for both gate and up slots.
                let lqt_gate = if absolute_offsets
                    .contains_key(&format!("blk.{}.ffn_gate.weight", layer_idx))
                {
                    t(&tensor_types, layer_idx, "ffn_gate.weight")
                } else {
                    lqt_up
                };
                compiled_layers.push(CompiledLayerEntry {
                    offsets,
                    blob_base_words,
                    quant_qk: lqt_main,
                    quant_v: lqt_v,
                    quant_attn_out: lqt_attn_out,
                    quant_ffn_down: lqt_down,
                    quant_ffn_gate: lqt_gate,
                    quant_ffn_up: lqt_up,
                });
                layer_idx += 1;
            }
            println!(
                "[Metadata] Compiled {} layers into lookup table.",
                compiled_layers.len()
            );
        }

        Ok(Self {
            version,
            tensor_count,
            tensor_offsets: absolute_offsets,
            tensor_types,
            tensor_dims,
            data_start_offset: data_start,
            gguf_metadata,
            compiled_layers,
        })
    }
    /// Construct ModelSpec from the parsed GGUF metadata
    pub fn to_model_spec(&self) -> ModelSpec {
        let mut spec = ModelSpec::from_gguf_metadata(&self.gguf_metadata);
        // Derive has_qk_norm from tensor presence (Qwen3: blk.0.attn_q_norm.weight / attn_k_norm.weight)
        let has_qk_norm = self.tensor_dims.contains_key("blk.0.attn_q_norm.weight")
            && self.tensor_dims.contains_key("blk.0.attn_k_norm.weight");
        // Derive post_norm_enabled from tensor presence (Gemma-2: blk.0.post_attention_norm.weight / post_ffw_norm.weight)
        let post_norm_enabled = self
            .tensor_dims
            .contains_key("blk.0.post_attention_norm.weight")
            && self.tensor_dims.contains_key("blk.0.post_ffw_norm.weight");
        // Set before compute_derived so arch-based fallback doesn't override
        spec.has_qk_norm = has_qk_norm;
        spec.post_norm_enabled = post_norm_enabled;

        // Derive packed-K stride from actual tensor shapes (not arch string).
        // For packed formats (Qwen3), attn_q/attn_k column counts differ from n_embd;
        // for standard models dims[1] == n_embd, so these equal the shader default (dim).
        if let Some(dims) = self.tensor_dims.get("blk.0.attn_q.weight") {
            if dims.len() >= 2 {
                spec.q_weight_k = dims[1] as usize;
            }
        }
        if let Some(dims) = self.tensor_dims.get("blk.0.attn_k.weight") {
            if dims.len() >= 2 {
                spec.k_weight_k = dims[1] as usize;
            }
        }

        // If head_dim was not in GGUF metadata (e.g. Qwen3 omits attention.key_length),
        // infer it from the Q weight shape: blk.0.attn_q.weight dims = [n_embd, n_head * head_dim]
        if spec.n_head.checked_div(spec.n_head).is_some() {
            // Try direct key first, then search for any blk.0.attn_q key
            let q_key = "blk.0.attn_q.weight";
            let dims_opt = self.tensor_dims.get(q_key).or_else(|| {
                self.tensor_dims
                    .keys()
                    .find(|k| k.contains("attn_q.weight") && k.starts_with("blk.0"))
                    .and_then(|k| self.tensor_dims.get(k))
            });
            if let Some(dims) = dims_opt {
                if dims.len() >= 2 {
                    let inferred = (dims[1] as usize) / spec.n_head;
                    if inferred > 0 && inferred != spec.head_dim {
                        eprintln!(
                            "[Spec] head_dim corrected {} -> {} via {} shape {:?}",
                            spec.head_dim, inferred, q_key, dims
                        );
                        spec.head_dim = inferred;
                        spec = spec.compute_derived();
                    }
                }
            }
        }
        spec
    }

    pub fn get_tensor_offset(&self, name: &str) -> Option<u64> {
        self.tensor_offsets.get(name).copied()
    }

    pub fn get_tensor_type(&self, name: &str) -> Option<u32> {
        self.tensor_types.get(name).copied()
    }

    pub fn get_layer_offsets(
        &self,
        layer_idx: usize,
        _model_arch: &str,
    ) -> Option<super::pipeline::LayerOffsets> {
        // e.g., "blk.0.attn_norm.weight"

        // Per-layer base byte: one word before the aligned minimum tensor offset.
        // Mirrors the compiled_layers loop so both paths produce base-relative
        // packed offsets that the shader reconstructs via blob_base_words.
        let min_offset = [
            "attn_norm.weight",
            "attn_norm.bias",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_qkv.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_norm.bias",
            "ffn_gate.weight",
            "ffn_down.weight",
            "ffn_up.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_q.bias",
            "attn_k.bias",
            "attn_v.bias",
        ]
        .iter()
        .filter_map(|s| {
            let key = format!("blk.{}.{}", layer_idx, s);
            self.tensor_offsets.get(&key).copied()
        })
        .filter(|&o| o > 0)
        .min()
        .unwrap_or(0);
        // Packed offset zero is reserved as the "tensor missing" sentinel.
        // Keep the first present tensor one word above the layer base.
        let base_byte = (min_offset & !3u64).saturating_sub(4);

        let rel = |abs: u64| -> u32 {
            if abs == 0 {
                0
            } else {
                super::pipeline::relative_packed_offset(abs, base_byte).expect("abs underflow")
            }
        };
        let p = |s: &str| -> u32 {
            let key = format!("blk.{}.{}", layer_idx, s);
            if let Some(&val) = self.tensor_offsets.get(&key) {
                return rel(val);
            }
            // Fused QKV fallback for Phi/GPT2/Other arch (GROUP C): use attn_qkv offset for q/k/v
            // so we don't panic on missing separate tensors. The fused layout is Q then K then V concatenated.
            if (s == "attn_q.weight" || s == "attn_k.weight" || s == "attn_v.weight")
                && self
                    .tensor_offsets
                    .contains_key(&format!("blk.{}.attn_qkv.weight", layer_idx))
            {
                return rel(self.tensor_offsets[&format!("blk.{}.attn_qkv.weight", layer_idx)]);
            }
            // Fused FFN gate_up for StarCoder2 etc (GROUP C)
            if (s == "ffn_gate.weight" || s == "ffn_up.weight")
                && self
                    .tensor_offsets
                    .contains_key(&format!("blk.{}.ffn_gate_up.weight", layer_idx))
            {
                return rel(self.tensor_offsets[&format!("blk.{}.ffn_gate_up.weight", layer_idx)]);
            }
            // Critical failure: layer exists but sub-tensor is missing
            panic!(
                "Layer {} exists but tensor '{}' is missing!",
                layer_idx, key
            );
        };

        // If primary weights are missing, return None (layer doesn't exist)
        self.tensor_offsets
            .get(&format!("blk.{}.attn_norm.weight", layer_idx))?;

        Some(super::pipeline::LayerOffsets {
            attn_norm: p("attn_norm.weight"),
            attn_norm_bias: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.attn_norm.bias", layer_idx))
                .copied()
                .unwrap_or(0)),
            attn_q: p("attn_q.weight"),
            attn_k: p("attn_k.weight"),
            attn_v: p("attn_v.weight"),
            attn_out: p("attn_output.weight"),
            ffn_norm: p("ffn_norm.weight"),
            ffn_norm_bias: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.ffn_norm.bias", layer_idx))
                .copied()
                .unwrap_or(0)),
            ffn_gate: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.ffn_gate.weight", layer_idx))
                .copied()
                .unwrap_or(0)),
            ffn_down: p("ffn_down.weight"),
            ffn_up: p("ffn_up.weight"),
            layer_idx: layer_idx as u32,
            attn_q_norm: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.attn_q_norm.weight", layer_idx))
                .copied()
                .unwrap_or(0)),
            attn_k_norm: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.attn_k_norm.weight", layer_idx))
                .copied()
                .unwrap_or(0)),
            attn_q_bias: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.attn_q.bias", layer_idx))
                .copied()
                .unwrap_or(0)),
            attn_k_bias: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.attn_k.bias", layer_idx))
                .copied()
                .unwrap_or(0)),
            attn_v_bias: rel(self
                .tensor_offsets
                .get(&format!("blk.{}.attn_v.bias", layer_idx))
                .copied()
                .unwrap_or(0)),
            // For Q4_K_M mixed quantization: determine if V and ffn_down are Q4_K or Q6_K
            v_is_q4k: self
                .tensor_types
                .get(&format!("blk.{}.attn_v.weight", layer_idx))
                .map(|&t| (t == 12) as u32)
                .unwrap_or(0),
            ffn_down_is_q4k: self
                .tensor_types
                .get(&format!("blk.{}.ffn_down.weight", layer_idx))
                .map(|&t| (t == 12) as u32)
                .unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_model_spec_head_dim_correction() {
        let mut gguf_metadata = HashMap::new();
        gguf_metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("qwen3".to_string()),
        );
        gguf_metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(16));
        gguf_metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(2048));
        gguf_metadata.insert("llama.block_count".to_string(), GgufValue::U32(28));
        gguf_metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(5632),
        );
        gguf_metadata.insert("llama.context_length".to_string(), GgufValue::U32(131072));

        let mut tensor_dims = HashMap::new();
        tensor_dims.insert("blk.0.attn_q.weight".to_string(), vec![2048, 2048]);

        let meta = BindlessMetadata {
            version: 3,
            tensor_count: 1,
            tensor_offsets: HashMap::new(),
            tensor_types: HashMap::new(),
            tensor_dims,
            data_start_offset: 0,
            gguf_metadata,
            compiled_layers: vec![],
        };
        let spec = meta.to_model_spec();
        // n_embd / n_head = 2048 / 16 = 128
        assert_eq!(spec.head_dim, 128);
    }

    #[test]
    fn test_get_layer_offsets_missing_layer_returns_none() {
        let meta = BindlessMetadata {
            version: 3,
            tensor_count: 0,
            tensor_offsets: HashMap::new(),
            tensor_types: HashMap::new(),
            tensor_dims: HashMap::new(),
            data_start_offset: 0,
            gguf_metadata: HashMap::new(),
            compiled_layers: vec![],
        };
        assert!(meta.get_layer_offsets(0, "llama").is_none());
    }

    #[test]
    fn test_get_layer_offsets_separate_qkv() {
        let mut tensor_offsets = HashMap::new();
        for (suffix, off) in [
            ("ffn_gate.weight", 100u64),
            ("attn_norm.weight", 200u64),
            ("attn_q.weight", 300u64),
            ("attn_k.weight", 400u64),
            ("attn_v.weight", 500u64),
            ("attn_output.weight", 600u64),
            ("ffn_norm.weight", 700u64),
            ("ffn_down.weight", 800u64),
            ("ffn_up.weight", 900u64),
        ] {
            tensor_offsets.insert(format!("blk.0.{suffix}"), off);
        }
        let mut tensor_types = HashMap::new();
        tensor_types.insert("blk.0.attn_q.weight".to_string(), 2u32);
        tensor_types.insert("blk.0.attn_v.weight".to_string(), 2u32);
        tensor_types.insert("blk.0.ffn_down.weight".to_string(), 12u32);

        let meta = BindlessMetadata {
            version: 3,
            tensor_count: 9,
            tensor_offsets,
            tensor_types,
            tensor_dims: HashMap::new(),
            data_start_offset: 0,
            gguf_metadata: HashMap::new(),
            compiled_layers: vec![],
        };
        let offs = meta.get_layer_offsets(0, "llama").expect("layer 0 exists");
        // The base is one aligned word before the minimum tensor offset so a
        // present ffn_gate never collides with the zero/missing sentinel.
        let base = 96;
        assert_eq!(
            offs.ffn_gate,
            super::super::pipeline::relative_packed_offset(100, base).unwrap()
        );
        assert_ne!(offs.ffn_gate, 0);
        assert_eq!(
            offs.attn_norm,
            super::super::pipeline::relative_packed_offset(200, base).unwrap()
        );
        assert_eq!(
            offs.attn_q,
            super::super::pipeline::relative_packed_offset(300, base).unwrap()
        );
        assert_eq!(
            offs.attn_k,
            super::super::pipeline::relative_packed_offset(400, base).unwrap()
        );
        assert_eq!(
            offs.attn_v,
            super::super::pipeline::relative_packed_offset(500, base).unwrap()
        );
        assert_eq!(
            offs.ffn_down,
            super::super::pipeline::relative_packed_offset(800, base).unwrap()
        );
        assert_eq!(offs.v_is_q4k, 0);
        assert_eq!(offs.ffn_down_is_q4k, 1);
    }

    #[test]
    fn test_quant_type_row_bytes_matches_fused_qkv_pattern() {
        // phi-4 merged ffn_up: Q4_K (12), in_dim 5120 => (5120/256)*144 = 2880.
        assert_eq!(quant_type_row_bytes(12, 5120), 2880);
        // F16 (1): in_dim * 2. F32 (0): in_dim * 4.
        assert_eq!(quant_type_row_bytes(1, 5120), 10240);
        assert_eq!(quant_type_row_bytes(0, 5120), 20480);
        // Q6_K (14): (cols/256)*210.
        assert_eq!(quant_type_row_bytes(14, 2560), 2100);
    }

    #[test]
    fn test_merged_gate_up_detection_shapes() {
        // GGUF dims are [in, out]. Merged SWIGLU (phi-3/4):
        //   ffn_up  = [n_embd, 2*ff_dim]  (5120, 35840)
        //   ffn_down = [ff_dim, n_embd]   (17920, 5120)
        // Detection: up_out (35840) == 2 * down_in (17920) AND no ffn_gate tensor.
        let up_out_dim = 35840u64;
        let down_in_dim = 17920u64;
        assert_eq!(up_out_dim, down_in_dim * 2, "phi-4 merged ffn_up shape");

        // Non-merged (tinyllama): ffn_up = [n_embd, ffn_dim], ffn_down = [ffn_dim, n_embd].
        let tl_up_out = 5632u64;
        let tl_down_in = 5632u64;
        assert_ne!(tl_up_out, tl_down_in * 2, "tinyllama ffn_up is NOT merged");

        // Row stride: ff_dim rows of the merged up tensor at 2880 B/row.
        let ff_dim = down_in_dim;
        let bpr = quant_type_row_bytes(12, 5120);
        assert_eq!(ff_dim * bpr, 17920 * 2880);
        assert_eq!(ff_dim * bpr, 51_609_600);
    }
}
