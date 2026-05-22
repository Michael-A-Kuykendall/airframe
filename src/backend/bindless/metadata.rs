use crate::core::spec::{GgufValue, ModelSpec};
use super::pipeline::CompiledLayerEntry;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

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

impl BindlessMetadata {
    /// scan a GGUF reader and extract tensor offsets.
    pub fn new<R: Read + Seek>(reader: &mut R) -> Self {
        let _start_pos = reader.stream_position().unwrap();

        // 1. Header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"GGUF", "Invalid Magic");

        let version = read_u32(reader);
        let tensor_count = read_u64(reader);
        let metadata_kv_count = read_u64(reader);

        // 2. Scan Metadata KVs — capture everything into gguf_metadata
        println!("[Metadata] Scanning {} KV pairs...", metadata_kv_count);
        let mut gguf_metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(reader);
            let val_type = read_u32(reader);

            let value = read_gguf_value(reader, val_type);

            // Debug log interesting keys
            match &value {
                GgufValue::U32(v)
                    if key.contains("head_count")
                        || key.contains("block_count")
                        || key.contains("embedding")
                        || key.contains("context_length")
                        || key.contains("feed_forward")
                        || key.contains("file_type") =>
                {
                    println!("[Metadata] {} = {}", key, v);
                }
                GgufValue::F32(v) if key.contains("epsilon") || key.contains("freq_base") => {
                    println!("[Metadata] {} = {}", key, v);
                }
                GgufValue::String(v)
                    if key.contains("architecture")
                        || key.contains("name")
                        || key.contains("model") =>
                {
                    println!("[Metadata] {} = {}", key, v);
                }
                _ => {}
            }

            gguf_metadata.insert(key, value);
        }

        // 3. Read Tensor Infos
        let mut tensor_offsets = HashMap::new();
        let mut tensor_types = HashMap::new();
        let mut tensor_dims = HashMap::new();

        for _ in 0..tensor_count {
            let name = read_string(reader);
            let n_dims = read_u32(reader);

            // Capture Dims
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(read_u64(reader));
            }

            let val_type = read_u32(reader); // ggml_type
            let offset = read_u64(reader); // relative data offset

            // Debug ALL tensors (Temporarily)
            println!(
                "[Metadata] Found {}: Type={} Dims={:?} Offset={}",
                name, val_type, dims, offset
            );

            tensor_offsets.insert(name.clone(), offset);
            tensor_types.insert(name.clone(), val_type);
            tensor_dims.insert(name, dims);
        }

        // 4. Alignment Padding
        // GGUF v3: data starts at aligned boundary.
        // Usually 32 bytes (llama.cpp default).
        // The spec says data_start is after tensor infos, aligned.
        let raw_end = reader.stream_position().unwrap();

        // We assume 32-byte alignment for now (safe bet for llama.cpp models)
        // Ideally we read `general.alignment` from metadata, but let's assume 32.
        let alignment = 32;
        let data_start = (raw_end + alignment - 1) & !(alignment - 1);

        // Adjust relative offsets to absolute
        // GGUF offsets are relative to `data_start`.
        // We want absolute file byte offsets for Bindless (or relative to data_start if we bind that view).
        // But Bindless binds the WHOLE file.
        // So absolute_offset = data_start + relative_offset.

        let mut absolute_offsets = HashMap::new();
        for (k, v) in tensor_offsets {
            absolute_offsets.insert(k, data_start + v);
        }

        // FSE compiled-layer table: single pass over layer indices at load time.
        // Eliminates per-token format!/HashMap overhead from the inference hot path.
        let mut compiled_layers = Vec::new();
        {
            let p = |offsets: &HashMap<String, u64>, layer: usize, s: &str| -> u32 {
                offsets.get(&format!("blk.{}.{}", layer, s))
                    .copied()
                    .unwrap_or(0) as u32
            };
            let t = |types: &HashMap<String, u32>, layer: usize, s: &str| -> u32 {
                types.get(&format!("blk.{}.{}", layer, s))
                    .copied()
                    .unwrap_or(2) // default Q4_0
            };

            let mut layer_idx = 0usize;
            while absolute_offsets.contains_key(&format!("blk.{}.attn_norm.weight", layer_idx)) {
                let offsets = super::pipeline::LayerOffsets {
                    attn_norm: p(&absolute_offsets, layer_idx, "attn_norm.weight"),
                    attn_q:    p(&absolute_offsets, layer_idx, "attn_q.weight"),
                    attn_k:    p(&absolute_offsets, layer_idx, "attn_k.weight"),
                    attn_v:    p(&absolute_offsets, layer_idx, "attn_v.weight"),
                    attn_out:  p(&absolute_offsets, layer_idx, "attn_output.weight"),
                    ffn_norm:  p(&absolute_offsets, layer_idx, "ffn_norm.weight"),
                    ffn_gate:  p(&absolute_offsets, layer_idx, "ffn_gate.weight"),
                    ffn_down:  p(&absolute_offsets, layer_idx, "ffn_down.weight"),
                    ffn_up:    p(&absolute_offsets, layer_idx, "ffn_up.weight"),
                    padding:   [layer_idx as u32, 0, 0],
                };
                let lqt_main = t(&tensor_types, layer_idx, "attn_q.weight");
                let lqt_v    = t(&tensor_types, layer_idx, "attn_v.weight");
                let lqt_down = t(&tensor_types, layer_idx, "ffn_down.weight");
                compiled_layers.push(CompiledLayerEntry {
                    offsets,
                    quant_type_packed: lqt_main | (lqt_v << 8) | (lqt_down << 16),
                });
                layer_idx += 1;
            }
            println!("[Metadata] Compiled {} layers into lookup table.", compiled_layers.len());
        }

        Self {
            version,
            tensor_count,
            tensor_offsets: absolute_offsets,
            tensor_types,
            tensor_dims,
            data_start_offset: data_start,
            gguf_metadata,
            compiled_layers,
        }
    }

    /// Construct ModelSpec from the parsed GGUF metadata
    pub fn to_model_spec(&self) -> ModelSpec {
        ModelSpec::from_gguf_metadata(&self.gguf_metadata)
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

        let p = |s: &str| -> u32 {
            let key = format!("blk.{}.{}", layer_idx, s);
            let val = self.tensor_offsets.get(&key);
            if val.is_none() {
                // Critical failure: layer exists but sub-tensor is missing
                panic!(
                    "Layer {} exists but tensor '{}' is missing!",
                    layer_idx, key
                );
            }
            *val.unwrap() as u32
        };

        // If primary weights are missing, return None (layer doesn't exist)
        if self
            .tensor_offsets
            .get(&format!("blk.{}.attn_norm.weight", layer_idx))
            .is_none()
        {
            return None;
        }

        Some(super::pipeline::LayerOffsets {
            attn_norm: p("attn_norm.weight"),
            attn_q: p("attn_q.weight"),
            attn_k: p("attn_k.weight"),
            attn_v: p("attn_v.weight"),
            attn_out: p("attn_output.weight"),
            ffn_norm: p("ffn_norm.weight"),
            ffn_gate: p("ffn_gate.weight"),
            ffn_down: p("ffn_down.weight"),
            ffn_up: p("ffn_up.weight"),
            padding: [layer_idx as u32, 0, 0], // padding[0] = layer_idx for shader
        })
    }
}

fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).unwrap();
    u32::from_le_bytes(buf)
}

fn read_u64<R: Read>(r: &mut R) -> u64 {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn read_string<R: Read>(r: &mut R) -> String {
    let len = read_u64(r);
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

fn read_gguf_value<R: Read + Seek>(r: &mut R, val_type: u32) -> GgufValue {
    match val_type {
        0 => {
            // uint8
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf).unwrap();
            GgufValue::U8(buf[0])
        }
        1 => {
            // int8
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf).unwrap();
            GgufValue::I8(buf[0] as i8)
        }
        2 => {
            // uint16
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf).unwrap();
            GgufValue::U16(u16::from_le_bytes(buf))
        }
        3 => {
            // int16
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf).unwrap();
            GgufValue::I16(i16::from_le_bytes(buf))
        }
        4 => {
            // uint32
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf).unwrap();
            GgufValue::U32(u32::from_le_bytes(buf))
        }
        5 => {
            // int32
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf).unwrap();
            GgufValue::I32(i32::from_le_bytes(buf))
        }
        6 => {
            // float32
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf).unwrap();
            GgufValue::F32(f32::from_le_bytes(buf))
        }
        7 => {
            // bool
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf).unwrap();
            GgufValue::Bool(buf[0] != 0)
        }
        8 => {
            // string
            GgufValue::String(read_string(r))
        }
        9 => {
            // array - skip contents, store length
            let item_type = read_u32(r);
            let len = read_u64(r);
            for _ in 0..len {
                skip_value(r, item_type);
            }
            GgufValue::ArrayLen(len as usize)
        }
        10 => {
            // uint64
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf).unwrap();
            GgufValue::U64(u64::from_le_bytes(buf))
        }
        11 => {
            // int64
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf).unwrap();
            GgufValue::I64(i64::from_le_bytes(buf))
        }
        12 => {
            // float64
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf).unwrap();
            GgufValue::F64(f64::from_le_bytes(buf))
        }
        // Malformed GGUF: unknown value type code; reader position is undefined — abort parse.
        _ => panic!("Unknown GGUF value type {}", val_type),
    }
}

fn skip_value<R: Read + Seek>(r: &mut R, val_type: u32) {
    match val_type {
        // 1 Byte
        0 | 1 | 7 => {
            // uint8, int8, bool
            r.seek(SeekFrom::Current(1)).unwrap();
        }
        // 2 Bytes
        2 | 3 => {
            // uint16, int16
            r.seek(SeekFrom::Current(2)).unwrap();
        }
        // 4 Bytes
        4 | 5 | 6 => {
            // uint32, int32, float32
            r.seek(SeekFrom::Current(4)).unwrap();
        }
        // 8 Bytes
        10 | 11 | 12 => {
            // uint64, int64, float64
            r.seek(SeekFrom::Current(8)).unwrap();
        }
        // String
        8 => {
            let len = read_u64(r);
            r.seek(SeekFrom::Current(len as i64)).unwrap();
        }
        // Array
        9 => {
            let item_type = read_u32(r);
            let len = read_u64(r);
            for _ in 0..len {
                skip_value(r, item_type);
            }
        }
        // Malformed GGUF: unknown type code; size unknown so reader position cannot be advanced.
        _ => panic!("Unknown GGUF value type {}", val_type),
    }
}
