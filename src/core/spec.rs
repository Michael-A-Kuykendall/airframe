//! Model architecture specification.
//!
//! Defines hyperparameters for transformer models. Supports auto-detection
//! from GGUF metadata or manual construction for known models.

use std::collections::HashMap;

/// GGUF file type IDs (general.file_type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufFileType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 8,
    Q5_1 = 9,
    Q8_0 = 7,
    Q2_K = 10,
    Q3_K = 11, // Q3_K_S in some listings
    Q4_K = 12, // Q4_K_S
    Q5_K = 13, // Q5_K_S
    Q6_K = 14,
    Unknown = 255,
}

impl From<u32> for GgufFileType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            7 => Self::Q8_0,
            8 => Self::Q5_0,
            9 => Self::Q5_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            _ => Self::Unknown,
        }
    }
}

/// Model architecture family (from general.architecture)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Mistral,
    Phi,
    Gemma,
    Other(String),
}

impl From<&str> for ModelArch {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "phi" | "phi2" | "phi3" => Self::Phi,
            "gemma" => Self::Gemma,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Transformer model hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    // Core dimensions
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub ff_dim: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rope_dim: usize,
    /// YaRN: low-frequency threshold. Dims with wavelength > L_train/alpha are not scaled.
    /// Standard default: 1.0 (scale all dims with wavelengths shorter than training context).
    pub yarn_alpha: f32,
    /// YaRN: high-frequency threshold. Dims with wavelength < L_train/beta get full linear scaling.
    /// Standard default: 32.0 (preserve high-frequency positional accuracy).
    pub yarn_beta: f32,
    pub n_ctx: usize,

    // Derived dimensions (computed once, used everywhere)
    pub head_dim: usize,  // n_embd / n_head
    pub gqa_ratio: usize, // n_head / n_head_kv
    pub kv_dim: usize,    // n_head_kv * head_dim

    // Architecture metadata
    pub arch: ModelArch,
    pub file_type: GgufFileType,
    pub model_name: String,

    // Buffer sizing (computed from dims)
    pub temp_buffer_size: usize, // max(n_embd, ff_dim*2 + n_embd) rounded up
    pub kv_cache_size_per_layer: usize, // n_ctx * n_head_kv * head_dim * 4 (F32)
}

impl ModelSpec {
    /// Compute derived fields from core dimensions
    pub fn compute_derived(mut self) -> Self {
        self.head_dim = self.n_embd / self.n_head;
        self.gqa_ratio = self.n_head / self.n_head_kv;
        self.kv_dim = self.n_head_kv * self.head_dim;

        // Temp buffer needs to hold the largest intermediate:
        // FFN uses ff_dim*2 (gate+up) plus dim for residual connections
        let ffn_scratch = self.ff_dim * 2 + self.n_embd;
        let min_scratch = std::cmp::max(self.n_embd * 4, ffn_scratch);
        // Round up to next 1024 boundary for GPU alignment
        self.temp_buffer_size = (min_scratch + 1023) & !1023;

        // KV cache: each layer stores K and V as F32
        // Size per K or V buffer = max_seq * kv_heads * head_dim * sizeof(f32)
        self.kv_cache_size_per_layer = self.n_ctx * self.n_head_kv * self.head_dim * 4;

        self
    }

    /// Construct ModelSpec from GGUF metadata key-value pairs.
    /// Keys follow GGUF standard: `{arch}.{param}` (e.g. "llama.embedding_length")
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Self {
        // Extract architecture
        let arch_str = match metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => s.clone(),
            _ => "llama".to_string(), // default
        };
        let arch = ModelArch::from(arch_str.as_str());
        let prefix = &arch_str; // "llama", "mistral", etc.

        // Extract model name
        let model_name = match metadata.get("general.name") {
            Some(GgufValue::String(s)) => s.clone(),
            _ => "unknown".to_string(),
        };

        // Extract file type
        let file_type = match metadata.get("general.file_type") {
            Some(GgufValue::U32(v)) => GgufFileType::from(*v),
            _ => GgufFileType::Unknown,
        };

        // Helper to get u32 metadata
        let get_u32 = |key: &str| -> Option<usize> {
            metadata.get(key).and_then(|v| match v {
                GgufValue::U32(n) => Some(*n as usize),
                _ => None,
            })
        };

        // Helper to get f32 metadata
        let get_f32 = |key: &str| -> Option<f32> {
            metadata.get(key).and_then(|v| match v {
                GgufValue::F32(n) => Some(*n),
                _ => None,
            })
        };

        // Extract vocab size from tokenizer tokens array length
        let n_vocab = match metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::ArrayLen(len)) => *len,
            _ => {
                // Fallback: try to read from model-specific key
                get_u32(&format!("{}.vocab_size", prefix)).unwrap_or(32000)
            }
        };

        let n_embd = get_u32(&format!("{}.embedding_length", prefix))
            .expect("Missing embedding_length in GGUF metadata");
        let n_layer = get_u32(&format!("{}.block_count", prefix))
            .expect("Missing block_count in GGUF metadata");
        let ff_dim = get_u32(&format!("{}.feed_forward_length", prefix))
            .expect("Missing feed_forward_length in GGUF metadata");
        let n_head = get_u32(&format!("{}.attention.head_count", prefix))
            .expect("Missing attention.head_count in GGUF metadata");
        let n_head_kv = get_u32(&format!("{}.attention.head_count_kv", prefix)).unwrap_or(n_head); // MHA fallback: n_head_kv == n_head
        let rms_eps =
            get_f32(&format!("{}.attention.layer_norm_rms_epsilon", prefix)).unwrap_or(1e-5);
        let rope_base = get_f32(&format!("{}.rope.freq_base", prefix)).unwrap_or(10000.0);
        let rope_dim =
            get_u32(&format!("{}.rope.dimension_count", prefix)).unwrap_or(n_embd / n_head); // default: head_dim
        let n_ctx = get_u32(&format!("{}.context_length", prefix)).unwrap_or(2048);

        Self {
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            ff_dim,
            rms_eps,
            rope_base,
            rope_scale: 1.0, // no standard GGUF key for this yet
            rope_dim,
            yarn_alpha: 1.0,
            yarn_beta: 32.0,
            n_ctx,
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch,
            file_type,
            model_name,
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
        }
        .compute_derived()
    }

    /// Expected spec for TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf
    pub fn tinylama_1_1b_chat_v1_0() -> Self {
        Self {
            n_vocab: 32000,
            n_embd: 2048,
            n_layer: 22,
            n_head: 32,
            n_head_kv: 4,
            ff_dim: 5632,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rope_dim: 64,
            yarn_alpha: 1.0,
            yarn_beta: 32.0,
            n_ctx: 2048,
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch: ModelArch::Llama,
            file_type: GgufFileType::Q4_0,
            model_name: "tinyllama_tinyllama-1.1b-chat-v1.0".to_string(),
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
        }
        .compute_derived()
    }

    /// Name for GGUF tensor lookup (blk.N.xxx for Llama-family)
    pub fn layer_prefix(&self) -> &str {
        match &self.arch {
            ModelArch::Llama | ModelArch::Mistral => "blk",
            ModelArch::Phi => "blk", // Phi also uses blk in GGUF
            ModelArch::Gemma => "blk",
            ModelArch::Other(_) => "blk", // default
        }
    }

    /// Architecture string for pipeline dispatch (e.g. "tinyllama", "llama", etc.)
    pub fn arch_string(&self) -> &str {
        match &self.arch {
            ModelArch::Llama => "llama",
            ModelArch::Mistral => "mistral",
            ModelArch::Phi => "phi",
            ModelArch::Gemma => "gemma",
            ModelArch::Other(s) => s.as_str(),
        }
    }

    /// GGML block size for the dominant quantization type
    pub fn quant_block_size(&self) -> usize {
        match self.file_type {
            GgufFileType::Q4_0 => 32,
            GgufFileType::Q4_1 => 32,
            GgufFileType::Q5_0 => 32,
            GgufFileType::Q5_1 => 32,
            GgufFileType::Q8_0 => 32,
            GgufFileType::Q2_K => 256,
            GgufFileType::Q3_K => 256,
            GgufFileType::Q4_K => 256,
            GgufFileType::Q5_K => 256,
            GgufFileType::Q6_K => 256,
            GgufFileType::F16 => 1,
            GgufFileType::F32 => 1,
            GgufFileType::Unknown => 32,
        }
    }

    /// Bytes per quantization block for the dominant type
    pub fn quant_block_bytes(&self) -> usize {
        match self.file_type {
            GgufFileType::Q4_0 => 18, // 2 (scale) + 16 (4-bit × 32 / 8)
            GgufFileType::Q4_1 => 20, // 2 (scale) + 2 (min) + 16
            GgufFileType::Q5_0 => 22, // 2 + 4 + 16
            GgufFileType::Q5_1 => 24, // 2 + 2 + 4 + 16
            GgufFileType::Q8_0 => 34, // 2 (scale) + 32 (8-bit × 32)
            GgufFileType::Q2_K => 84,
            GgufFileType::Q3_K => 110,
            GgufFileType::Q4_K => 144,
            GgufFileType::Q5_K => 176,
            GgufFileType::Q6_K => 210,
            GgufFileType::F16 => 2,
            GgufFileType::F32 => 4,
            GgufFileType::Unknown => 18,
        }
    }

    /// Row byte size for a weight matrix in the dominant quant type
    pub fn quant_row_bytes(&self, cols: usize) -> usize {
        (cols / self.quant_block_size()) * self.quant_block_bytes()
    }
}

/// Parsed GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// For array types, we store just the length (we don't need token arrays in ModelSpec)
    ArrayLen(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tinyllama_derived() {
        let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        assert_eq!(spec.head_dim, 64);
        assert_eq!(spec.gqa_ratio, 8);
        assert_eq!(spec.kv_dim, 256);
        assert_eq!(spec.kv_cache_size_per_layer, 2048 * 4 * 64 * 4);
        assert!(spec.temp_buffer_size >= 5632 * 2 + 2048);
    }

    #[test]
    fn test_from_gguf_metadata() {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        meta.insert(
            "general.name".to_string(),
            GgufValue::String("test_model".to_string()),
        );
        meta.insert("general.file_type".to_string(), GgufValue::U32(2));
        meta.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        meta.insert("llama.block_count".to_string(), GgufValue::U32(32));
        meta.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        meta.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(32),
        );
        meta.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::F32(1e-5),
        );
        meta.insert("llama.rope.freq_base".to_string(), GgufValue::F32(10000.0));
        meta.insert(
            "llama.rope.dimension_count".to_string(),
            GgufValue::U32(128),
        );
        meta.insert("llama.context_length".to_string(), GgufValue::U32(4096));
        meta.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayLen(32000),
        );

        let spec = ModelSpec::from_gguf_metadata(&meta);
        assert_eq!(spec.n_embd, 4096);
        assert_eq!(spec.n_layer, 32);
        assert_eq!(spec.ff_dim, 11008);
        assert_eq!(spec.n_head, 32);
        assert_eq!(spec.n_head_kv, 32);
        assert_eq!(spec.head_dim, 128);
        assert_eq!(spec.gqa_ratio, 1); // MHA
        assert_eq!(spec.n_vocab, 32000);
        assert_eq!(spec.n_ctx, 4096);
        assert!(matches!(spec.arch, ModelArch::Llama));
        assert!(matches!(spec.file_type, GgufFileType::Q4_0));
    }

    #[test]
    fn test_quant_row_bytes_q4_0() {
        let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        // dim=2048: 2048/32 * 18 = 1152 bytes per row
        assert_eq!(spec.quant_row_bytes(2048), 1152);
    }
}
