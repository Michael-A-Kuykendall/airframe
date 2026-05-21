use crate::core::dequant::{dequantize_q4_k, dequantize_q6_k};
use crate::core::ggml_types::{ggml_type_bytes_per_tensor, ggml_type_name, validate_tensor_bounds};
use crate::core::{
    error::{LibshimmyError, Result},
    spec::ModelSpec,
    tensor::Tensor,
    weight_id::WeightId,
};
use crate::ensure;
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Model container holding FP32 tensors keyed by WeightId
#[derive(Debug)]
pub struct Model {
    pub spec: ModelSpec,
    pub weights: HashMap<WeightId, Tensor>,
}

/// GGUF tensor metadata
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

/// GGUF file header
#[derive(Debug)]
struct GgufHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

impl Model {
    pub fn new(spec: ModelSpec) -> Self {
        Self {
            spec,
            weights: HashMap::new(),
        }
    }

    pub fn insert_weight(&mut self, id: WeightId, tensor: Tensor) {
        self.weights.insert(id, tensor);
    }

    pub fn get_weight(&self, id: &WeightId) -> Option<&Tensor> {
        self.weights.get(id)
    }

    /// Load TinyLlama Q4_0 GGUF model and dequantize to FP32
    /// This is a v0-specific implementation targeting the exact model
    pub fn from_tinylama_q4_0_gguf<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Some GGUF exporters omit `general.alignment`. llama.cpp historically used 32-byte
        // alignment as the default for tensor data. For the TinyLlama Q4_0 target model,
        // we allow an explicit fallback to 32 when the key is missing.
        const TINYLAMA_Q4_0_FALLBACK_ALIGNMENT: u64 = 32;

        let model =
            Self::from_gguf_with_alignment_fallback(path, Some(TINYLAMA_Q4_0_FALLBACK_ALIGNMENT))?;
        ensure!(
            model.spec == ModelSpec::tinylama_1_1b_chat_v1_0(),
            "GGUF metadata-derived ModelSpec did not match TinyLlama expected spec: got={:?}",
            model.spec
        );
        Ok(model)
    }

    /// Load a GGUF model and dequantize weights to FP32.
    ///
    /// Step 3 requirement: `ModelSpec` is derived from GGUF metadata.
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_gguf_with_alignment_fallback(path, None)
    }

    fn from_gguf_with_alignment_fallback<P: AsRef<Path>>(
        path: P,
        alignment_fallback: Option<u64>,
    ) -> Result<Self> {
        let file = std::fs::File::open(&path).map_err(LibshimmyError::Io)?;

        let file_len = file.metadata().map_err(LibshimmyError::Io)?.len();

        println!(
            "📁 File length: {} bytes ({:.2} MB)",
            file_len,
            file_len as f64 / 1024.0 / 1024.0
        );

        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LibshimmyError::Io(std::io::Error::other(format!("Failed to mmap file: {}", e)))
        })?;

        let mut cursor = std::io::Cursor::new(&mmap[..]);

        // Parse GGUF header
        let header = parse_gguf_header(&mut cursor)?;
        println!(
            "📋 Header: version={}, tensor_count={}, metadata_kv_count={}",
            header.version, header.tensor_count, header.metadata_kv_count
        );

        // Parse GGUF metadata (alignment + ModelSpec derivation)
        let metadata_start_pos = cursor.position();
        let metadata = parse_metadata(&mut cursor, header.metadata_kv_count)?;
        let metadata_end_pos = cursor.position();
        println!(
            "📝 Metadata: {} bytes (pos {} -> {}), keys={} ",
            metadata_end_pos - metadata_start_pos,
            metadata_start_pos,
            metadata_end_pos,
            metadata.len()
        );

        if std::env::var("SHIMMY_DUMP_GGUF_METADATA").ok().as_deref() == Some("1") {
            dump_metadata_keys(&metadata);
        }

        // Parse tensor info section with comprehensive instrumentation
        let tensor_info_start_pos = cursor.position();
        println!("\n📊 COMPREHENSIVE GGUF ANALYSIS:");
        println!(
            "📁 File length: {} bytes ({:.2} MB)",
            file_len,
            file_len as f64 / 1024.0 / 1024.0
        );
        println!(
            "📋 Header: version={}, tensor_count={}, metadata_kv_count={}",
            header.version, header.tensor_count, header.metadata_kv_count
        );
        println!("📝 Metadata section: {} bytes", tensor_info_start_pos - 20);
        println!(
            "🔍 Tensor info starts at position: {}",
            tensor_info_start_pos
        );

        // Check what's at this position
        let mut peek_bytes = [0u8; 32];
        let bytes_read = cursor.read(&mut peek_bytes).map_err(LibshimmyError::Io)?;
        println!(
            "👀 Next {} bytes: {:02x?}",
            bytes_read,
            &peek_bytes[..bytes_read]
        );

        // Reset position and try to parse tensor info directly

        // Fix: Parse exactly metadata_kv_count (23) entries to maintain cursor alignment
        cursor.set_position(tensor_info_start_pos);

        println!("🔍 Parsing tensor infos sequentially after metadata (no scanning needed)");

        // Parse tensor infos directly - they should be right after metadata
        let tensor_infos =
            parse_tensor_infos_with_validation(&mut cursor, header.tensor_count, file_len)?;

        // CRITICAL: Get the tensor data base offset WITH ALIGNMENT
        // GGUF spec: tensor data section starts at next aligned boundary after tensor infos
        let alignment = get_general_alignment(&metadata, alignment_fallback)?;
        let raw_offset = cursor.position();
        let tensor_data_base_offset = align_up(raw_offset, alignment)?;
        println!("🎯 Raw offset after tensor infos: {}", raw_offset);
        println!(
            "🎯 Tensor data starts at file offset: {} (aligned to {})",
            tensor_data_base_offset, alignment
        );

        let spec = model_spec_from_metadata(&metadata)?;
        println!("🧩 ModelSpec from metadata: {:?}", spec);
        let mut model = Model::new(spec);

        // Load and dequantize required weights
        let n_layer = model.spec.n_layer;
        load_required_weights(
            &mut model,
            &tensor_infos,
            &mmap,
            tensor_data_base_offset,
            n_layer,
        )?;

        // Validate all required weights are present
        validate_required_weights(&model, n_layer)?;

        Ok(model)
    }
}

/// Parse GGUF file header
fn parse_gguf_header<R: Read>(reader: &mut R) -> Result<GgufHeader> {
    // Check magic number "GGUF"
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(LibshimmyError::Io)?;

    if &magic != b"GGUF" {
        return Err(LibshimmyError::FixtureError {
            msg: "Invalid GGUF magic number".to_string(),
        });
    }

    let version = reader
        .read_u32::<LittleEndian>()
        .map_err(LibshimmyError::Io)?;

    if version != 3 {
        return Err(LibshimmyError::FixtureError {
            msg: format!("Unsupported GGUF version: {}", version),
        });
    }

    let tensor_count = reader
        .read_u64::<LittleEndian>()
        .map_err(LibshimmyError::Io)?;

    let metadata_kv_count = reader
        .read_u64::<LittleEndian>()
        .map_err(LibshimmyError::Io)?;

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Parse metadata section with complete GGUF value type support
#[allow(dead_code)]
fn skip_metadata<R: Read + Seek>(reader: &mut R, kv_count: u64) -> Result<()> {
    println!(
        "📝 Parsing {} metadata key-value pairs with complete GGUF support...",
        kv_count
    );

    let start_pos = reader.stream_position().map_err(LibshimmyError::Io)?;
    let mut parsed_count = 0;

    for i in 0..kv_count {
        let pos_before = reader.stream_position().map_err(LibshimmyError::Io)?;

        // Try to read key length
        let key_len_result = reader.read_u64::<LittleEndian>();

        if let Ok(key_len) = key_len_result {
            if key_len == 0 {
                println!(
                    "✅ Found zero key length at KV {}, this indicates start of tensor section",
                    i
                );
                reader
                    .seek(SeekFrom::Start(pos_before))
                    .map_err(LibshimmyError::Io)?;
                break;
            }

            if key_len > 1000 {
                println!("⚠️  Unreasonable key length {} at KV {}", key_len, i);
                println!(
                    "    Raw bytes at pos {}: {:02x?}",
                    pos_before,
                    &key_len.to_le_bytes()
                );

                // This is likely misalignment from a previous array - let's see what's actually here
                reader
                    .seek(SeekFrom::Start(pos_before))
                    .map_err(LibshimmyError::Io)?;
                let mut debug_bytes = [0u8; 64];
                let bytes_read = reader.read(&mut debug_bytes).unwrap_or(0);
                println!(
                    "    Next {} bytes: {:02x?}",
                    bytes_read,
                    &debug_bytes[..bytes_read]
                );

                return Err(LibshimmyError::FixtureError {
                    msg: format!("Misaligned metadata parsing at KV {} - likely array parsing error in previous KV", i),
                });
            }

            // Try to read the key
            let mut key_bytes = vec![0u8; key_len as usize];
            if reader.read_exact(&mut key_bytes).is_err() {
                println!(
                    "⚠️  Failed to read key at KV {}, assuming we've reached tensor section",
                    i
                );
                reader
                    .seek(SeekFrom::Start(pos_before))
                    .map_err(LibshimmyError::Io)?;
                break;
            }

            let key_name = String::from_utf8_lossy(&key_bytes);

            // Try to read value type
            let pos_before_type = reader.stream_position().map_err(LibshimmyError::Io)?;
            println!("KV {}: key_len={} at pos={}", i, key_len, pos_before);
            println!("key='{}' value_type at pos={}", key_name, pos_before_type);

            // Peek at the next 16 bytes to see what we're about to read
            let mut peek_bytes = [0u8; 16];
            let bytes_read = reader.read(&mut peek_bytes).unwrap_or(0);
            println!(
                "next {} bytes: {:02x?}",
                bytes_read,
                &peek_bytes[..bytes_read]
            );
            reader
                .seek(SeekFrom::Start(pos_before_type))
                .map_err(LibshimmyError::Io)?;

            let value_type_result = reader.read_u32::<LittleEndian>();
            if let Ok(value_type) = value_type_result {
                println!(
                    "    {} value_type={}",
                    match value_type {
                        4 => "u32",
                        6 => "u64",
                        8 => "string",
                        10 => "f32",
                        _ => "unknown",
                    },
                    value_type
                );
                // Parse value with complete GGUF type support
                let skip_result = (|| -> Result<()> {
                    match value_type {
                        0 => {
                            // u8
                            reader
                                .seek(SeekFrom::Current(1))
                                .map_err(LibshimmyError::Io)?;
                        }
                        1 => {
                            // i8
                            reader
                                .seek(SeekFrom::Current(1))
                                .map_err(LibshimmyError::Io)?;
                        }
                        2 => {
                            // u16
                            reader
                                .seek(SeekFrom::Current(2))
                                .map_err(LibshimmyError::Io)?;
                        }
                        3 => {
                            // i16
                            reader
                                .seek(SeekFrom::Current(2))
                                .map_err(LibshimmyError::Io)?;
                        }
                        4 => {
                            // u32
                            reader
                                .seek(SeekFrom::Current(4))
                                .map_err(LibshimmyError::Io)?;
                        }
                        5 => {
                            // i32
                            reader
                                .seek(SeekFrom::Current(4))
                                .map_err(LibshimmyError::Io)?;
                        }
                        6 => {
                            // u64 - but handle special cases where it's actually f32
                            if key_name == "llama.attention.layer_norm_rms_epsilon"
                                || key_name == "llama.rope.freq_base"
                            {
                                // These are actually f32 but encoded as type 6 in this GGUF file
                                let f32_val = reader
                                    .read_f32::<LittleEndian>()
                                    .map_err(LibshimmyError::Io)?;
                                println!("    📊 {} f32 value: {}", key_name, f32_val);
                                // No additional padding needed - it's just 4 bytes
                            } else {
                                reader
                                    .seek(SeekFrom::Current(8))
                                    .map_err(LibshimmyError::Io)?;
                            }
                        }
                        7 => {
                            // i64
                            reader
                                .seek(SeekFrom::Current(8))
                                .map_err(LibshimmyError::Io)?;
                        }
                        8 => {
                            // String
                            let str_len = reader
                                .read_u64::<LittleEndian>()
                                .map_err(LibshimmyError::Io)?;
                            reader
                                .seek(SeekFrom::Current(str_len as i64))
                                .map_err(LibshimmyError::Io)?;
                        }
                        9 => {
                            // Array
                            let array_type = reader
                                .read_u32::<LittleEndian>()
                                .map_err(LibshimmyError::Io)?;
                            let array_len = reader
                                .read_u64::<LittleEndian>()
                                .map_err(LibshimmyError::Io)?;

                            println!("    📊 Array: type={}, len={}", array_type, array_len);

                            match array_type {
                                0 => {
                                    reader.seek(SeekFrom::Current(array_len as i64))?;
                                } // u8 array
                                1 => {
                                    reader.seek(SeekFrom::Current(array_len as i64))?;
                                } // i8 array
                                2 => {
                                    reader.seek(SeekFrom::Current((array_len * 2) as i64))?;
                                } // u16 array
                                3 => {
                                    reader.seek(SeekFrom::Current((array_len * 2) as i64))?;
                                } // i16 array
                                4 => {
                                    reader.seek(SeekFrom::Current((array_len * 4) as i64))?;
                                } // u32 array
                                5 => {
                                    reader.seek(SeekFrom::Current((array_len * 4) as i64))?;
                                } // i32 array
                                6 => {
                                    // u64 array - but might be f32 in some cases
                                    if key_name == "tokenizer.ggml.scores" {
                                        // This is actually f32 array but encoded as u64 array
                                        reader.seek(SeekFrom::Current((array_len * 4) as i64))?;
                                    } else {
                                        reader.seek(SeekFrom::Current((array_len * 8) as i64))?;
                                    }
                                }
                                7 => {
                                    reader.seek(SeekFrom::Current((array_len * 8) as i64))?;
                                } // i64 array
                                8 => {
                                    // String array
                                    for _ in 0..array_len {
                                        let elem_str_len = reader
                                            .read_u64::<LittleEndian>()
                                            .map_err(LibshimmyError::Io)?;
                                        reader
                                            .seek(SeekFrom::Current(elem_str_len as i64))
                                            .map_err(LibshimmyError::Io)?;
                                    }
                                }
                                10 => {
                                    reader.seek(SeekFrom::Current((array_len * 4) as i64))?;
                                } // f32 array
                                11 => {
                                    reader.seek(SeekFrom::Current(array_len as i64))?;
                                } // bool array
                                12 => {
                                    reader.seek(SeekFrom::Current((array_len * 8) as i64))?;
                                } // f64 array
                                _ => {
                                    return Err(LibshimmyError::FixtureError {
                                        msg: format!(
                                            "Unsupported array element type: {}",
                                            array_type
                                        ),
                                    });
                                }
                            }
                        }
                        10 => {
                            // f32
                            reader
                                .seek(SeekFrom::Current(4))
                                .map_err(LibshimmyError::Io)?;
                        }
                        11 => {
                            // bool
                            reader
                                .seek(SeekFrom::Current(1))
                                .map_err(LibshimmyError::Io)?;
                        }
                        12 => {
                            // f64
                            reader
                                .seek(SeekFrom::Current(8))
                                .map_err(LibshimmyError::Io)?;
                        }
                        _ => {
                            return Err(LibshimmyError::FixtureError {
                                msg: format!("Unsupported GGUF value type: {}", value_type),
                            });
                        }
                    }
                    Ok(())
                })();

                if skip_result.is_ok() {
                    let pos_after = reader.stream_position().map_err(LibshimmyError::Io)?;
                    println!(
                        "KV {} complete: pos {} -> {} ({} bytes)",
                        i,
                        pos_before,
                        pos_after,
                        pos_after - pos_before
                    );
                    parsed_count += 1;
                } else {
                    return Err(LibshimmyError::FixtureError {
                        msg: format!(
                            "Failed to parse metadata KV {} '{}': {:?}",
                            i,
                            key_name,
                            skip_result.err()
                        ),
                    });
                }
            } else {
                return Err(LibshimmyError::FixtureError {
                    msg: format!("Failed to read value type for metadata KV {}", i),
                });
            }
        } else {
            return Err(LibshimmyError::FixtureError {
                msg: format!("Failed to read key length for metadata KV {}", i),
            });
        }
    }

    // Hard invariant: we must parse exactly kv_count entries
    if parsed_count != kv_count {
        return Err(LibshimmyError::FixtureError {
            msg: format!(
                "Metadata parsing failed: expected {} KV pairs, parsed {}",
                kv_count, parsed_count
            ),
        });
    }

    let end_pos = reader.stream_position().map_err(LibshimmyError::Io)?;
    println!(
        "✅ Successfully parsed all {} metadata entries: {} bytes (pos {} -> {})",
        parsed_count,
        end_pos - start_pos,
        start_pos,
        end_pos
    );
    Ok(())
}

/// Parse tensor information with comprehensive validation and instrumentation
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum GgufMetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array { elem_type: u32, len: u64 },
    U64(u64),
    I64(i64),
    F64(f64),
}

fn dump_metadata_keys(metadata: &HashMap<String, GgufMetaValue>) {
    let mut keys: Vec<_> = metadata.keys().cloned().collect();
    keys.sort();
    println!("\n🧾 GGUF metadata keys ({}):", keys.len());
    for k in keys {
        println!("  {k}: {:?}", metadata.get(&k));
    }
}

fn model_spec_from_metadata(metadata: &HashMap<String, GgufMetaValue>) -> Result<ModelSpec> {
    let n_layer = require_usize(metadata, "llama.block_count")?;
    let n_embd = require_usize(metadata, "llama.embedding_length")?;
    let n_head = require_usize(metadata, "llama.attention.head_count")?;
    let n_head_kv = require_usize(metadata, "llama.attention.head_count_kv")?;
    let ff_dim = require_usize(metadata, "llama.feed_forward_length")?;
    let n_ctx = require_usize(metadata, "llama.context_length")?;
    let rope_base = require_f32(metadata, "llama.rope.freq_base")?;
    let rope_dim = require_usize(metadata, "llama.rope.dimension_count")?;
    let rms_eps = require_f32(metadata, "llama.attention.layer_norm_rms_epsilon")?;

    // Vocab size: prefer explicit metadata if present, otherwise infer from tokenizer token array length.
    let n_vocab_from_llama = optional_usize(metadata, "llama.vocab_size")?;
    let n_vocab_from_tokens = optional_array_len(metadata, "tokenizer.ggml.tokens")?;

    let n_vocab = match (n_vocab_from_llama, n_vocab_from_tokens) {
        (Some(v1), Some(v2)) => {
            ensure!(
                v1 == v2,
                "Vocab size mismatch: llama.vocab_size={} tokenizer.ggml.tokens.len={}",
                v1,
                v2
            );
            v1
        }
        (Some(v), None) => v,
        (None, Some(v)) => v,
        (None, None) => {
            return Err(LibshimmyError::InvariantViolation {
                msg: "Missing tokenizer vocab size: expected 'llama.vocab_size' or 'tokenizer.ggml.tokens'"
                    .to_string(),
            });
        }
    };

    // Step 4 tightens RoPE scaling rules; for now, the no-scaling case is represented by 1.0.
    let rope_scale = 1.0f32;

    ensure!(n_head > 0, "n_head must be > 0");
    ensure!(n_head_kv > 0, "n_head_kv must be > 0");
    ensure!(
        n_embd % n_head == 0,
        "n_embd % n_head must be 0 (n_embd={} n_head={})",
        n_embd,
        n_head
    );
    ensure!(
        n_head % n_head_kv == 0,
        "n_head % n_head_kv must be 0 (n_head={} n_head_kv={})",
        n_head,
        n_head_kv
    );

    Ok(ModelSpec {
        n_vocab,
        n_embd,
        n_layer,
        n_head,
        n_head_kv,
        ff_dim,
        rms_eps,
        rope_base,
        rope_scale,
        rope_dim,
        yarn_alpha: 1.0,
        yarn_beta: 32.0,
        n_ctx,
        head_dim: 0,
        gqa_ratio: 0,
        kv_dim: 0,
        arch: crate::core::spec::ModelArch::Llama,
        file_type: crate::core::spec::GgufFileType::Unknown,
        model_name: String::new(),
        temp_buffer_size: 0,
        kv_cache_size_per_layer: 0,
    }
    .compute_derived())
}

fn require_usize(metadata: &HashMap<String, GgufMetaValue>, key: &str) -> Result<usize> {
    optional_usize(metadata, key)?.ok_or_else(|| LibshimmyError::InvariantViolation {
        msg: format!("Missing required GGUF metadata key: {key}"),
    })
}

fn optional_usize(metadata: &HashMap<String, GgufMetaValue>, key: &str) -> Result<Option<usize>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(GgufMetaValue::U32(v)) => Ok(Some(*v as usize)),
        Some(GgufMetaValue::U64(v)) => Ok(Some((*v).try_into().map_err(|_| {
            LibshimmyError::InvariantViolation {
                msg: format!("{key} out of range for usize: {v}"),
            }
        })?)),
        Some(other) => Err(LibshimmyError::InvariantViolation {
            msg: format!("{key} has unexpected type: {other:?}"),
        }),
    }
}

fn require_f32(metadata: &HashMap<String, GgufMetaValue>, key: &str) -> Result<f32> {
    match metadata.get(key) {
        Some(GgufMetaValue::F32(v)) => Ok(*v),
        Some(GgufMetaValue::F64(v)) => Err(LibshimmyError::InvariantViolation {
            msg: format!("{key} is f64; expected f32: {v}"),
        }),
        Some(other) => Err(LibshimmyError::InvariantViolation {
            msg: format!("{key} has unexpected type: {other:?}"),
        }),
        None => Err(LibshimmyError::InvariantViolation {
            msg: format!("Missing required GGUF metadata key: {key}"),
        }),
    }
}

fn optional_array_len(
    metadata: &HashMap<String, GgufMetaValue>,
    key: &str,
) -> Result<Option<usize>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(GgufMetaValue::Array { len, .. }) => Ok(Some((*len).try_into().map_err(|_| {
            LibshimmyError::InvariantViolation {
                msg: format!("{key} len out of range for usize: {len}"),
            }
        })?)),
        Some(other) => Err(LibshimmyError::InvariantViolation {
            msg: format!("{key} has unexpected type: {other:?}"),
        }),
    }
}

fn parse_metadata<R: Read + Seek>(
    reader: &mut R,
    kv_count: u64,
) -> Result<HashMap<String, GgufMetaValue>> {
    let mut out = HashMap::new();
    for kv_idx in 0..kv_count {
        let key = read_gguf_string(reader).map_err(|e| LibshimmyError::FixtureError {
            msg: format!("Failed to read metadata key at KV {kv_idx}: {e}"),
        })?;
        let value_type = reader
            .read_u32::<LittleEndian>()
            .map_err(LibshimmyError::Io)?;

        let value = read_gguf_value(reader, value_type).map_err(|e| LibshimmyError::FixtureError {
            msg: format!("Failed to read metadata value for key '{key}' (type {value_type}) at KV {kv_idx}: {e}")
        })?;

        out.insert(key, value);
    }
    Ok(out)
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(LibshimmyError::Io)?;
    ensure!(
        len <= (1024 * 1024),
        "GGUF string length too large: {}",
        len
    );
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).map_err(LibshimmyError::Io)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn skip_gguf_string<R: Read + Seek>(reader: &mut R) -> Result<()> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(LibshimmyError::Io)?;
    ensure!(
        len <= (1024 * 1024),
        "GGUF string length too large: {}",
        len
    );
    reader
        .seek(SeekFrom::Current(len as i64))
        .map_err(LibshimmyError::Io)?;
    Ok(())
}

fn read_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<GgufMetaValue> {
    // GGUF v3 type codes (gguf.h):
    // 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool 8=string 9=array 10=u64 11=i64 12=f64
    Ok(match value_type {
        0 => GgufMetaValue::U8(reader.read_u8().map_err(LibshimmyError::Io)?),
        1 => GgufMetaValue::I8(reader.read_i8().map_err(LibshimmyError::Io)?),
        2 => GgufMetaValue::U16(
            reader
                .read_u16::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        3 => GgufMetaValue::I16(
            reader
                .read_i16::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        4 => GgufMetaValue::U32(
            reader
                .read_u32::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        5 => GgufMetaValue::I32(
            reader
                .read_i32::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        6 => GgufMetaValue::F32(
            reader
                .read_f32::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        7 => {
            let b = reader.read_u8().map_err(LibshimmyError::Io)?;
            ensure!(b == 0 || b == 1, "Invalid GGUF bool value: {}", b);
            GgufMetaValue::Bool(b == 1)
        }
        8 => GgufMetaValue::String(read_gguf_string(reader)?),
        9 => {
            let elem_type = reader
                .read_u32::<LittleEndian>()
                .map_err(LibshimmyError::Io)?;
            let len = reader
                .read_u64::<LittleEndian>()
                .map_err(LibshimmyError::Io)?;
            ensure!(elem_type != 9, "Nested GGUF arrays unsupported");
            match elem_type {
                0 | 1 | 7 => {
                    reader
                        .seek(SeekFrom::Current(len as i64))
                        .map_err(LibshimmyError::Io)?;
                }
                2 | 3 => {
                    let bytes =
                        len.checked_mul(2)
                            .ok_or_else(|| LibshimmyError::InvariantViolation {
                                msg: format!(
                                    "array byte length overflow: elem_type={elem_type} len={len}"
                                ),
                            })?;
                    reader
                        .seek(SeekFrom::Current(bytes as i64))
                        .map_err(LibshimmyError::Io)?;
                }
                4 | 5 | 6 => {
                    let bytes =
                        len.checked_mul(4)
                            .ok_or_else(|| LibshimmyError::InvariantViolation {
                                msg: format!(
                                    "array byte length overflow: elem_type={elem_type} len={len}"
                                ),
                            })?;
                    reader
                        .seek(SeekFrom::Current(bytes as i64))
                        .map_err(LibshimmyError::Io)?;
                }
                8 => {
                    for _ in 0..len {
                        skip_gguf_string(reader)?;
                    }
                }
                10 | 11 | 12 => {
                    let bytes =
                        len.checked_mul(8)
                            .ok_or_else(|| LibshimmyError::InvariantViolation {
                                msg: format!(
                                    "array byte length overflow: elem_type={elem_type} len={len}"
                                ),
                            })?;
                    reader
                        .seek(SeekFrom::Current(bytes as i64))
                        .map_err(LibshimmyError::Io)?;
                }
                other => {
                    return Err(LibshimmyError::Unsupported(format!(
                        "Unsupported GGUF array element type: {other}"
                    )));
                }
            }
            GgufMetaValue::Array { elem_type, len }
        }
        10 => GgufMetaValue::U64(
            reader
                .read_u64::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        11 => GgufMetaValue::I64(
            reader
                .read_i64::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        12 => GgufMetaValue::F64(
            reader
                .read_f64::<LittleEndian>()
                .map_err(LibshimmyError::Io)?,
        ),
        other => {
            return Err(LibshimmyError::Unsupported(format!(
                "Unsupported GGUF metadata value type: {}",
                other
            )))
        }
    })
}

fn get_general_alignment(
    metadata: &HashMap<String, GgufMetaValue>,
    alignment_fallback: Option<u64>,
) -> Result<u64> {
    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetaValue::U32(v)) => *v as u64,
        Some(GgufMetaValue::U64(v)) => *v,
        Some(other) => {
            return Err(LibshimmyError::InvariantViolation {
                msg: format!("general.alignment has unexpected type: {other:?}"),
            });
        }
        None => {
            if let Some(fallback) = alignment_fallback {
                println!(
                    "⚠️  Missing GGUF metadata key general.alignment; using explicit fallback alignment={}.",
                    fallback
                );
                fallback
            } else {
                println!("⚠️  Missing GGUF metadata key general.alignment; defaulting to 32.");
                32
            }
        }
    };

    ensure!(alignment > 0, "general.alignment must be > 0");
    ensure!(
        alignment.is_power_of_two(),
        "general.alignment must be power-of-two, got {}",
        alignment
    );
    ensure!(
        alignment <= 256,
        "general.alignment too large (max 256), got {}",
        alignment
    );
    Ok(alignment)
}

fn align_up(offset: u64, alignment: u64) -> Result<u64> {
    ensure!(alignment > 0, "alignment must be > 0");
    ensure!(
        alignment.is_power_of_two(),
        "alignment must be power-of-two, got {}",
        alignment
    );
    Ok((offset + alignment - 1) & !(alignment - 1))
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

    #[test]
    fn test_align_up_basic() {
        assert_eq!(align_up(0, 32).unwrap(), 0);
        assert_eq!(align_up(1, 32).unwrap(), 32);
        assert_eq!(align_up(31, 32).unwrap(), 32);
        assert_eq!(align_up(32, 32).unwrap(), 32);
        assert_eq!(align_up(33, 32).unwrap(), 64);
    }

    #[test]
    fn test_align_up_rejects_invalid_alignment() {
        let err = align_up(123, 24).unwrap_err();
        match err {
            LibshimmyError::InvariantViolation { msg } => {
                assert!(msg.contains("power-of-two"));
            }
            other => panic!("expected InvariantViolation, got {other:?}"),
        }
    }
}

/// Parse tensor information with comprehensive validation and instrumentation
fn parse_tensor_infos_with_validation<R: Read>(
    reader: &mut R,
    tensor_count: u64,
    file_len: u64,
) -> Result<Vec<GgufTensorInfo>> {
    println!("\n📊 TENSOR ANALYSIS:");
    println!(
        "{:<4} {:<30} {:<15} {:<10} {:<15} {:<15} {:<10}",
        "#", "Name", "Shape", "Type", "Offset", "Size (bytes)", "End Pos"
    );
    println!("{}", "-".repeat(100));

    let mut tensor_infos = Vec::new();

    for i in 0..tensor_count {
        // Read tensor name
        let name_len = reader.read_u64::<LittleEndian>().map_err(|e| {
            println!("❌ Failed to read tensor {} name length: {}", i, e);
            LibshimmyError::Io(e)
        })?;

        ensure!(
            name_len <= 1000,
            "Unreasonable tensor name length {} for tensor {}",
            name_len,
            i
        );

        let mut name_bytes = vec![0u8; name_len as usize];
        reader.read_exact(&mut name_bytes).map_err(|e| {
            println!("❌ Failed to read tensor {} name: {}", i, e);
            LibshimmyError::Io(e)
        })?;

        let name = String::from_utf8(name_bytes).map_err(|e| LibshimmyError::FixtureError {
            msg: format!("Invalid tensor name UTF-8: {}", e),
        })?;

        // Read dimensions
        let n_dims = reader.read_u32::<LittleEndian>().map_err(|e| {
            println!("❌ Failed to read tensor {} dimension count: {}", i, e);
            LibshimmyError::Io(e)
        })?;

        let mut dimensions = Vec::new();
        for _ in 0..n_dims {
            let dim = reader.read_u64::<LittleEndian>().map_err(|e| {
                println!("❌ Failed to read tensor {} dimension: {}", i, e);
                LibshimmyError::Io(e)
            })?;
            dimensions.push(dim as usize);
        }

        // Read GGML type
        let ggml_type = reader.read_u32::<LittleEndian>().map_err(|e| {
            println!("❌ Failed to read tensor {} GGML type: {}", i, e);
            LibshimmyError::Io(e)
        })?;

        // Read offset
        let offset = reader.read_u64::<LittleEndian>().map_err(|e| {
            println!("❌ Failed to read tensor {} offset: {}", i, e);
            LibshimmyError::Io(e)
        })?;

        // Calculate tensor size based on type and dimensions with overflow protection
        let element_count: usize = dimensions.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| LibshimmyError::FixtureError {
                    msg: format!("Dimension overflow for tensor '{}': {:?}", name, dimensions),
                })
        })?;

        // Use fail-closed type system - no fallback to FP32 assumptions
        let tensor_size_bytes =
            ggml_type_bytes_per_tensor(ggml_type, element_count).map_err(|e| match e {
                LibshimmyError::QuantUnsupported {
                    ggml_type,
                    type_name,
                    ..
                } => LibshimmyError::QuantUnsupported {
                    tensor_name: name.clone(),
                    ggml_type,
                    type_name,
                },
                other => other,
            })?;

        let end_pos = offset + tensor_size_bytes as u64;

        // Print tensor info
        println!(
            "{:<4} {:<30} {:<15} {:<10} {:<15} {:<15} {:<10}",
            i,
            if name.len() > 30 {
                format!("{}...", &name[..27])
            } else {
                name.clone()
            },
            format!("{:?}", dimensions),
            ggml_type,
            offset,
            tensor_size_bytes,
            end_pos
        );

        // CRITICAL CHECK: Fail immediately if tensor extends beyond file
        ensure!(
            end_pos <= file_len,
            "Tensor '{}' extends beyond file: end_pos={} > file_len={}",
            name,
            end_pos,
            file_len
        );

        tensor_infos.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
        });
    }

    println!("\n✅ All {} tensor layouts are valid", tensor_count);
    Ok(tensor_infos)
}

fn load_tensor_by_type(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    match tensor_info.ggml_type {
        0 => {
            // F32: raw little-endian floats
            let total_elements: usize = tensor_info.dimensions.iter().product();
            let byte_len =
                total_elements
                    .checked_mul(4)
                    .ok_or_else(|| LibshimmyError::FixtureError {
                        msg: format!(
                            "F32 byte length overflow for tensor '{}': {:?}",
                            tensor_info.name, tensor_info.dimensions
                        ),
                    })?;
            let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
            let data_end = data_start + byte_len;
            ensure!(
                data_end <= mmap.len(),
                "Tensor '{}' extends beyond file (F32): end={} > file_len={}",
                tensor_info.name,
                data_end,
                mmap.len()
            );

            let mut out = Vec::with_capacity(total_elements);
            let bytes = &mmap[data_start..data_end];
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Tensor::new(out, tensor_info.dimensions.clone())
        }
        2 => dequantize_q4_0(tensor_info, mmap, tensor_data_base_offset),
        6 => crate::core::dequant::dequantize_q5_0(tensor_info, mmap, tensor_data_base_offset),
        8 => crate::core::dequant::dequantize_q8_0(tensor_info, mmap, tensor_data_base_offset),
        12 => dequantize_q4_k(tensor_info, mmap, tensor_data_base_offset),
        14 => dequantize_q6_k(tensor_info, mmap, tensor_data_base_offset),
        other => Err(LibshimmyError::QuantUnsupported {
            tensor_name: tensor_info.name.clone(),
            ggml_type: other,
            type_name: format!("UNKNOWN_{}", other),
        }),
    }
}

/// Load and dequantize required weights
fn load_required_weights(
    model: &mut Model,
    tensor_infos: &[GgufTensorInfo],
    mmap: &Mmap,
    tensor_data_base_offset: u64,
    n_layer: usize,
) -> Result<()> {
    let weight_mapping = create_weight_mapping(n_layer);

    for tensor_info in tensor_infos {
        if let Some(weight_id) = weight_mapping.get(&tensor_info.name) {
            let fp32_tensor = load_tensor_by_type(tensor_info, mmap, tensor_data_base_offset)?;
            model.insert_weight(weight_id.clone(), fp32_tensor);
        }
    }

    Ok(())
}

fn create_weight_mapping(n_layer: usize) -> HashMap<String, WeightId> {
    let mut mapping = HashMap::new();

    // Token embeddings
    mapping.insert("token_embd.weight".to_string(), WeightId::TokenEmbed);

    // Output projection
    mapping.insert("output.weight".to_string(), WeightId::OutputProj);

    // Layer weights
    for layer in 0..n_layer {
        // Attention weights
        mapping.insert(
            format!("blk.{}.attn_q.weight", layer),
            WeightId::AttnQ { layer },
        );
        mapping.insert(
            format!("blk.{}.attn_k.weight", layer),
            WeightId::AttnK { layer },
        );
        mapping.insert(
            format!("blk.{}.attn_v.weight", layer),
            WeightId::AttnV { layer },
        );
        mapping.insert(
            format!("blk.{}.attn_output.weight", layer),
            WeightId::AttnO { layer },
        );

        // FFN weights
        mapping.insert(
            format!("blk.{}.ffn_gate.weight", layer),
            WeightId::FfnGate { layer },
        );
        mapping.insert(
            format!("blk.{}.ffn_up.weight", layer),
            WeightId::FfnUp { layer },
        );
        mapping.insert(
            format!("blk.{}.ffn_down.weight", layer),
            WeightId::FfnDown { layer },
        );

        // Normalization weights
        mapping.insert(
            format!("blk.{}.attn_norm.weight", layer),
            WeightId::AttnNorm { layer },
        );
        mapping.insert(
            format!("blk.{}.ffn_norm.weight", layer),
            WeightId::FfnNorm { layer },
        );
    }

    // Final norm
    mapping.insert("output_norm.weight".to_string(), WeightId::OutputNorm);

    mapping
}

/// Dequantize Q4_0 tensor to FP32
fn dequantize_q4_0(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    crate::core::dequant::dequantize_q4_0(tensor_info, mmap, tensor_data_base_offset)
}

/// Validate all required weights are present
fn validate_required_weights(model: &Model, n_layer: usize) -> Result<()> {
    let required_weights = get_required_weights(n_layer);

    for weight_id in required_weights {
        if model.get_weight(&weight_id).is_none() {
            return Err(LibshimmyError::WeightMissing {
                weight_id: format!("{:?}", weight_id),
            });
        }
    }

    Ok(())
}

/// Get list of all required weights for TinyLlama
fn get_required_weights(n_layer: usize) -> Vec<WeightId> {
    WeightId::all_for_layers(n_layer)
}

/// Validate tensor layout against file size
#[allow(dead_code)]
fn validate_tensor_layout(tensor_infos: &[GgufTensorInfo], file_len: u64) -> Result<()> {
    println!("🔍 Validating tensor layout against file size...");

    for (i, tensor_info) in tensor_infos.iter().enumerate() {
        let element_count: usize = tensor_info.dimensions.iter().product();

        // Use fail-closed type system - no "assume FP32" fallback
        let tensor_size_bytes = ggml_type_bytes_per_tensor(tensor_info.ggml_type, element_count)
            .map_err(|e| match e {
                LibshimmyError::QuantUnsupported {
                    ggml_type,
                    type_name,
                    ..
                } => LibshimmyError::QuantUnsupported {
                    tensor_name: tensor_info.name.clone(),
                    ggml_type,
                    type_name,
                },
                other => other,
            })?;

        let tensor_end = tensor_info.offset + tensor_size_bytes as u64;

        // Get type name for logging
        let type_name = ggml_type_name(tensor_info.ggml_type).unwrap_or("UNKNOWN");

        println!(
            "📊 Tensor {}: '{}' type={} ({}) offset={} size={} end={} (file_len={})",
            i,
            tensor_info.name,
            tensor_info.ggml_type,
            type_name,
            tensor_info.offset,
            tensor_size_bytes,
            tensor_end,
            file_len
        );

        // Use the centralized bounds validation
        validate_tensor_bounds(
            &tensor_info.name,
            tensor_info.ggml_type,
            tensor_size_bytes,
            tensor_info.offset,
            file_len,
        )?;
    }

    println!("✅ All tensor layouts are valid");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let spec = ModelSpec::tinylama_1_1b_chat_v1_0();

        let model = Model::new(spec);
        assert_eq!(model.weights.len(), 0);
        assert_eq!(model.spec.n_vocab, 32000);
    }

    #[test]
    fn test_weight_insertion_and_retrieval() {
        let spec = ModelSpec {
            n_vocab: 1000,
            n_embd: 512,
            n_layer: 2,
            n_head: 8,
            n_head_kv: 2,
            ff_dim: 1024,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rope_dim: 64,
            n_ctx: 1024,
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch: crate::core::spec::ModelArch::Llama,
            file_type: crate::core::spec::GgufFileType::Unknown,
            model_name: String::new(),
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
        }
        .compute_derived();

        let mut model = Model::new(spec);

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let weight_id = WeightId::TokenEmbed;

        model.insert_weight(weight_id.clone(), tensor);

        let retrieved = model.get_weight(&weight_id).unwrap();
        assert_eq!(retrieved.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_weight_mapping_creation() {
        let mapping = create_weight_mapping(22);

        // Check token embedding
        assert_eq!(
            mapping.get("token_embd.weight"),
            Some(&WeightId::TokenEmbed)
        );

        // Check output projection
        assert_eq!(mapping.get("output.weight"), Some(&WeightId::OutputProj));

        // Check layer 0 attention weights
        assert_eq!(
            mapping.get("blk.0.attn_q.weight"),
            Some(&WeightId::AttnQ { layer: 0 })
        );
        assert_eq!(
            mapping.get("blk.0.attn_k.weight"),
            Some(&WeightId::AttnK { layer: 0 })
        );

        // Check layer 21 (last layer) weights
        assert_eq!(
            mapping.get("blk.21.ffn_norm.weight"),
            Some(&WeightId::FfnNorm { layer: 21 })
        );

        // Should have correct total count: 1 token + 1 output + 22*9 layer weights + 1 output_norm = 201
        assert_eq!(mapping.len(), 1 + 1 + 22 * 9 + 1);
    }

    #[test]
    fn test_required_weights_list() {
        let weights = get_required_weights(22);

        // Should match expected count
        assert_eq!(weights.len(), 1 + 22 * 9 + 2); // token + layers + output_norm + output

        // Check key weights are present
        assert!(weights.contains(&WeightId::TokenEmbed));
        assert!(weights.contains(&WeightId::OutputProj));
        assert!(weights.contains(&WeightId::OutputNorm));
        assert!(weights.contains(&WeightId::AttnQ { layer: 0 }));
        assert!(weights.contains(&WeightId::FfnNorm { layer: 21 }));
    }

    #[test]
    fn test_model_spec_from_metadata_happy_path() {
        let mut md = HashMap::new();

        md.insert("llama.block_count".to_string(), GgufMetaValue::U32(22));
        md.insert(
            "llama.embedding_length".to_string(),
            GgufMetaValue::U32(2048),
        );
        md.insert(
            "llama.attention.head_count".to_string(),
            GgufMetaValue::U32(32),
        );
        md.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufMetaValue::U32(4),
        );
        md.insert(
            "llama.feed_forward_length".to_string(),
            GgufMetaValue::U32(5632),
        );
        md.insert("llama.context_length".to_string(), GgufMetaValue::U32(2048));
        md.insert(
            "llama.rope.freq_base".to_string(),
            GgufMetaValue::F32(10000.0),
        );
        md.insert(
            "llama.rope.dimension_count".to_string(),
            GgufMetaValue::U32(64),
        );
        md.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            GgufMetaValue::F32(1e-5),
        );

        md.insert("llama.vocab_size".to_string(), GgufMetaValue::U32(32000));
        md.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetaValue::Array {
                elem_type: 8,
                len: 32000,
            },
        );

        let spec = model_spec_from_metadata(&md).unwrap();
        // model_spec_from_metadata doesn't know file_type or model_name (those come from
        // general.file_type / general.name which aren't in the test metadata), so build
        // the expected value from the canonical constructor and clear those fields.
        let mut expected = ModelSpec::tinylama_1_1b_chat_v1_0();
        expected.file_type = crate::core::spec::GgufFileType::Unknown;
        expected.model_name = String::new();
        assert_eq!(spec, expected);
    }

    #[test]
    fn test_model_spec_from_metadata_rejects_inconsistent_dims() {
        let mut md = HashMap::new();
        md.insert("llama.block_count".to_string(), GgufMetaValue::U32(22));
        md.insert(
            "llama.embedding_length".to_string(),
            GgufMetaValue::U32(2049),
        );
        md.insert(
            "llama.attention.head_count".to_string(),
            GgufMetaValue::U32(32),
        );
        md.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufMetaValue::U32(4),
        );
        md.insert(
            "llama.feed_forward_length".to_string(),
            GgufMetaValue::U32(5632),
        );
        md.insert("llama.context_length".to_string(), GgufMetaValue::U32(2048));
        md.insert(
            "llama.rope.freq_base".to_string(),
            GgufMetaValue::F32(10000.0),
        );
        md.insert(
            "llama.rope.dimension_count".to_string(),
            GgufMetaValue::U32(64),
        );
        md.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            GgufMetaValue::F32(1e-5),
        );
        md.insert("llama.vocab_size".to_string(), GgufMetaValue::U32(32000));

        let err = model_spec_from_metadata(&md).unwrap_err();
        match err {
            LibshimmyError::InvariantViolation { msg } => {
                assert!(msg.contains("n_embd % n_head"));
            }
            other => panic!("expected InvariantViolation, got {other:?}"),
        }
    }
}
