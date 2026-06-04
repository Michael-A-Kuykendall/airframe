//! Model architecture specification.
//!
//! Defines hyperparameters for transformer models. Supports auto-detection
//! from GGUF metadata or manual construction for known models.

use std::collections::HashMap;

/// GGUF file type IDs (general.file_type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// non_camel_case_types: GGUF spec uses Q4_0 etc. naming convention
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
    /// Qwen2 / Qwen2.5 (no per-head QK norm)
    Qwen2,
    /// Qwen3 (has per-head Q and K RMSNorm before RoPE)
    Qwen3,
    Other(String),
}

impl From<&str> for ModelArch {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "phi" | "phi2" | "phi3" => Self::Phi,
            "gemma" => Self::Gemma,
            "qwen2" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
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
    /// Attention logit soft-cap (Gemma-2: 50.0, others: 0.0 = disabled)
    pub attn_logit_softcap: f32,
    /// Final logit soft-cap applied after lm_head (Gemma-2: 30.0, others: 0.0 = disabled)
    pub final_logit_softcap: f32,
    /// Per-head Q and K RMSNorm before RoPE (Qwen3). False for all other architectures.
    pub has_qk_norm: bool,

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
        if self.head_dim == 0 {
            self.head_dim = self.n_embd / self.n_head;
        }
        // RoPE must not exceed head_dim — cap it (handles Gemma-2: key_length=256, rope_dim may default to n_embd/n_head=288)
        if self.rope_dim > self.head_dim {
            self.rope_dim = self.head_dim;
        }
        self.gqa_ratio = self.n_head / self.n_head_kv;
        self.kv_dim = self.n_head_kv * self.head_dim;
        // Qwen3 uses per-head Q and K RMSNorm before RoPE
        self.has_qk_norm = matches!(self.arch, ModelArch::Qwen3);

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
    ///
    /// FSE (Hit #2): single pass over the map — match by key suffix after the first `.`.
    /// GGUF suffixes are globally unique within a file (only one arch is present).
    /// Eliminates ~15 individual format!/HashMap::get calls with one O(N) scan.
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Self {
        let mut arch_str        = "llama".to_string();
        let mut model_name      = "unknown".to_string();
        let mut file_type_raw: Option<u32>   = None;
        let mut n_vocab:        Option<usize> = None;
        let mut n_embd:         Option<usize> = None;
        let mut n_layer:        Option<usize> = None;
        let mut ff_dim:         Option<usize> = None;
        let mut n_head:         Option<usize> = None;
        let mut n_head_kv:      Option<usize> = None;
        let mut rms_eps:        Option<f32>   = None;
        let mut rope_base:      Option<f32>   = None;
        let mut rope_dim:       Option<usize> = None;
        let mut n_ctx:          Option<usize> = None;
        let mut attn_softcap:   Option<f32>   = None;
        let mut final_softcap:  Option<f32>   = None;
        let mut head_dim_expl:  Option<usize> = None;

        // Single pass: dispatch on exact key for `general.*` / `tokenizer.*`,
        // or on the suffix after the first `.` for arch-prefixed keys.
        for (key, value) in metadata.iter() {
            match key.as_str() {
                "general.architecture" => {
                    if let GgufValue::String(s) = value { arch_str = s.clone(); }
                }
                "general.name" => {
                    if let GgufValue::String(s) = value { model_name = s.clone(); }
                }
                "general.file_type" => {
                    if let GgufValue::U32(v) = value { file_type_raw = Some(*v); }
                }
                "tokenizer.ggml.tokens" => {
                    if let GgufValue::ArrayLen(len) = value { n_vocab = Some(*len); }
                }
                _ => {
                    // Strip the arch prefix (everything up to and including the first '.')
                    // GGUF suffixes are unique per file; no collision risk.
                    let suffix = match key.find('.') {
                        Some(i) => &key[i + 1..],
                        None    => continue,
                    };
                    match suffix {
                        "embedding_length"                  => { if let GgufValue::U32(v) = value { n_embd       = Some(*v as usize); } }
                        "block_count"                       => { if let GgufValue::U32(v) = value { n_layer      = Some(*v as usize); } }
                        "feed_forward_length"               => { if let GgufValue::U32(v) = value { ff_dim       = Some(*v as usize); } }
                        "attention.head_count"              => { if let GgufValue::U32(v) = value { n_head       = Some(*v as usize); } }
                        "attention.head_count_kv"           => { if let GgufValue::U32(v) = value { n_head_kv    = Some(*v as usize); } }
                        "attention.layer_norm_rms_epsilon"
                        | "attention.layer_norm_epsilon"
                        | "layer_norm_epsilon"              => { if let GgufValue::F32(v) = value { rms_eps      = Some(*v); } }
                        "rope.freq_base"                    => { if let GgufValue::F32(v) = value { rope_base    = Some(*v); } }
                        "rope.dimension_count"              => { if let GgufValue::U32(v) = value { rope_dim     = Some(*v as usize); } }
                        "context_length"                    => { if let GgufValue::U32(v) = value { n_ctx        = Some(*v as usize); } }
                        "attn_logit_softcapping"
                        | "attention.logit_softcapping"     => { if let GgufValue::F32(v) = value { attn_softcap = Some(*v); } }
                        "final_logit_softcapping"           => { if let GgufValue::F32(v) = value { final_softcap = Some(*v); } }
                        "attention.key_length"              => { if let GgufValue::U32(v) = value { head_dim_expl = Some(*v as usize); } }
                        "vocab_size" => {
                            if n_vocab.is_none() {
                                if let GgufValue::U32(v) = value { n_vocab = Some(*v as usize); }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let arch      = ModelArch::from(arch_str.as_str());
        let file_type = file_type_raw.map(GgufFileType::from).unwrap_or(GgufFileType::Unknown);
        let n_embd    = n_embd.expect("Missing embedding_length in GGUF metadata");
        let n_layer   = n_layer.expect("Missing block_count in GGUF metadata");
        let ff_dim    = ff_dim.expect("Missing feed_forward_length in GGUF metadata");
        let n_head    = n_head.expect("Missing attention.head_count in GGUF metadata");
        let n_head_kv = n_head_kv.unwrap_or(n_head);
        let n_vocab   = n_vocab.unwrap_or(32000);

        Self {
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            ff_dim,
            rms_eps:             rms_eps.unwrap_or(1e-5),
            rope_base:           rope_base.unwrap_or(10000.0),
            rope_scale:          1.0,
            rope_dim:            rope_dim.unwrap_or(n_embd / n_head),
            yarn_alpha:          1.0,
            yarn_beta:           32.0,
            n_ctx:               n_ctx.unwrap_or(2048),
            attn_logit_softcap:  attn_softcap.unwrap_or(0.0),
            final_logit_softcap: final_softcap.unwrap_or(0.0),
            has_qk_norm:         false, // set in compute_derived() from arch
            head_dim:            head_dim_expl.unwrap_or(0),
            gqa_ratio:           0,
            kv_dim:              0,
            arch,
            file_type,
            model_name,
            temp_buffer_size:         0,
            kv_cache_size_per_layer:  0,
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
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            has_qk_norm: false,
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
            ModelArch::Qwen2 | ModelArch::Qwen3 => "blk",
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
            ModelArch::Qwen2 => "qwen2",
            ModelArch::Qwen3 => "qwen3",
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

    /// Whether this architecture should run LayerNorm math (mean+variance)
    /// instead of RMSNorm in bindless kernels and final norm.
    pub fn uses_layer_norm(&self) -> bool {
        match &self.arch {
            ModelArch::Phi => true,
            ModelArch::Other(s) => {
                let a = s.to_ascii_lowercase();
                a.contains("gpt2") || a.contains("starcoder") || a.contains("falcon")
            }
            _ => false,
        }
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

    // ── GgufFileType::from(u32) ───────────────────────────────────────────────

    #[test]
    fn test_gguf_file_type_all_known_variants() {
        assert!(matches!(GgufFileType::from(0), GgufFileType::F32));
        assert!(matches!(GgufFileType::from(1), GgufFileType::F16));
        assert!(matches!(GgufFileType::from(2), GgufFileType::Q4_0));
        assert!(matches!(GgufFileType::from(3), GgufFileType::Q4_1));
        assert!(matches!(GgufFileType::from(7), GgufFileType::Q8_0));
        assert!(matches!(GgufFileType::from(8), GgufFileType::Q5_0));
        assert!(matches!(GgufFileType::from(9), GgufFileType::Q5_1));
        assert!(matches!(GgufFileType::from(10), GgufFileType::Q2_K));
        assert!(matches!(GgufFileType::from(11), GgufFileType::Q3_K));
        assert!(matches!(GgufFileType::from(12), GgufFileType::Q4_K));
        assert!(matches!(GgufFileType::from(13), GgufFileType::Q5_K));
        assert!(matches!(GgufFileType::from(14), GgufFileType::Q6_K));
    }

    #[test]
    fn test_gguf_file_type_unknown_variant() {
        assert!(matches!(GgufFileType::from(99), GgufFileType::Unknown));
        assert!(matches!(GgufFileType::from(255), GgufFileType::Unknown));
        assert!(matches!(GgufFileType::from(4), GgufFileType::Unknown));
    }

    // ── ModelArch::from(&str) ─────────────────────────────────────────────────

    #[test]
    fn test_model_arch_all_known_variants() {
        assert!(matches!(ModelArch::from("llama"), ModelArch::Llama));
        assert!(matches!(ModelArch::from("mistral"), ModelArch::Mistral));
        assert!(matches!(ModelArch::from("phi"), ModelArch::Phi));
        assert!(matches!(ModelArch::from("phi2"), ModelArch::Phi));
        assert!(matches!(ModelArch::from("phi3"), ModelArch::Phi));
        assert!(matches!(ModelArch::from("gemma"), ModelArch::Gemma));
    }

    #[test]
    fn test_model_arch_case_insensitive() {
        assert!(matches!(ModelArch::from("LLAMA"), ModelArch::Llama));
        assert!(matches!(ModelArch::from("Mistral"), ModelArch::Mistral));
        assert!(matches!(ModelArch::from("PHI2"), ModelArch::Phi));
    }

    #[test]
    fn test_model_arch_other() {
        let a = ModelArch::from("starcoder2");
        match &a {
            ModelArch::Other(s) => assert_eq!(s, "starcoder2"),
            _ => panic!("expected Other"),
        }
    }

    // ── ModelSpec::arch_string ─────────────────────────────────────────────────

    #[test]
    fn test_arch_string_all_known() {
        let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        assert_eq!(spec.arch_string(), "llama");
    }

    #[test]
    fn test_arch_string_other() {
        let mut spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        spec.arch = ModelArch::Other("starcoder2".to_string());
        assert_eq!(spec.arch_string(), "starcoder2");
    }

    #[test]
    fn test_arch_string_gemma() {
        let mut spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        spec.arch = ModelArch::Gemma;
        assert_eq!(spec.arch_string(), "gemma");
    }

    // ── compute_derived: rope_dim capping ────────────────────────────────────

    #[test]
    fn test_compute_derived_caps_rope_dim() {
        let spec = ModelSpec {
            n_vocab: 1000,
            n_embd: 8,
            n_layer: 1,
            n_head: 2,
            n_head_kv: 2,
            ff_dim: 16,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rope_dim: 999, // way over head_dim=4
            yarn_alpha: 1.0,
            yarn_beta: 32.0,
            n_ctx: 128,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            has_qk_norm: false,
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch: ModelArch::Llama,
            file_type: GgufFileType::F32,
            model_name: "test".to_string(),
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
        }
        .compute_derived();
        assert_eq!(spec.head_dim, 4); // 8/2=4
        assert_eq!(spec.rope_dim, 4, "rope_dim should be capped to head_dim");
    }

    // ── quant_block_size / quant_block_bytes — all variants ──────────────────

    #[test]
    fn test_quant_block_size_all_variants() {
        let mut spec = ModelSpec::tinylama_1_1b_chat_v1_0();

        for (ft, expected_size) in [
            (GgufFileType::Q4_0, 32),
            (GgufFileType::Q4_1, 32),
            (GgufFileType::Q5_0, 32),
            (GgufFileType::Q5_1, 32),
            (GgufFileType::Q8_0, 32),
            (GgufFileType::Q2_K, 256),
            (GgufFileType::Q3_K, 256),
            (GgufFileType::Q4_K, 256),
            (GgufFileType::Q5_K, 256),
            (GgufFileType::Q6_K, 256),
            (GgufFileType::F16, 1),
            (GgufFileType::F32, 1),
            (GgufFileType::Unknown, 32),
        ] {
            spec.file_type = ft;
            assert_eq!(spec.quant_block_size(), expected_size, "{ft:?}");
        }
    }

    #[test]
    fn test_quant_block_bytes_all_variants() {
        let mut spec = ModelSpec::tinylama_1_1b_chat_v1_0();

        for (ft, expected_bytes) in [
            (GgufFileType::Q4_0, 18),
            (GgufFileType::Q4_1, 20),
            (GgufFileType::Q5_0, 22),
            (GgufFileType::Q5_1, 24),
            (GgufFileType::Q8_0, 34),
            (GgufFileType::Q2_K, 84),
            (GgufFileType::Q3_K, 110),
            (GgufFileType::Q4_K, 144),
            (GgufFileType::Q5_K, 176),
            (GgufFileType::Q6_K, 210),
            (GgufFileType::F16, 2),
            (GgufFileType::F32, 4),
            (GgufFileType::Unknown, 18),
        ] {
            spec.file_type = ft;
            assert_eq!(spec.quant_block_bytes(), expected_bytes, "{ft:?}");
        }
    }

    // ── from_gguf_metadata: edge cases ───────────────────────────────────────

    #[test]
    fn test_from_gguf_metadata_gqa_llama32() {
        // Llama-3.2: 16 heads, 8 kv heads → gqa_ratio=2
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
        meta.insert("general.name".to_string(), GgufValue::String("llama32".to_string()));
        meta.insert("general.file_type".to_string(), GgufValue::U32(12));
        meta.insert("llama.embedding_length".to_string(), GgufValue::U32(2048));
        meta.insert("llama.block_count".to_string(), GgufValue::U32(16));
        meta.insert("llama.feed_forward_length".to_string(), GgufValue::U32(8192));
        meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));
        meta.insert("llama.attention.layer_norm_rms_epsilon".to_string(), GgufValue::F32(1e-5));
        meta.insert("llama.rope.freq_base".to_string(), GgufValue::F32(500000.0));
        meta.insert("llama.rope.dimension_count".to_string(), GgufValue::U32(64));
        meta.insert("llama.context_length".to_string(), GgufValue::U32(131072));
        meta.insert("tokenizer.ggml.tokens".to_string(), GgufValue::ArrayLen(32000));

        let spec = ModelSpec::from_gguf_metadata(&meta);
        assert_eq!(spec.gqa_ratio, 4, "32/8 = 4");
        assert_eq!(spec.kv_dim, 8 * 64); // 8 kv heads × head_dim=64
    }

    #[test]
    fn test_from_gguf_metadata_arch_prefix_used_for_keys() {
        // Verify the suffix-scan picks up arch-prefixed keys
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), GgufValue::String("llama".to_string()));
        meta.insert("general.name".to_string(), GgufValue::String("x".to_string()));
        meta.insert("general.file_type".to_string(), GgufValue::U32(2));
        meta.insert("llama.embedding_length".to_string(), GgufValue::U32(512));
        meta.insert("llama.block_count".to_string(), GgufValue::U32(4));
        meta.insert("llama.feed_forward_length".to_string(), GgufValue::U32(1024));
        meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(8));
        meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));
        meta.insert("llama.attention.layer_norm_rms_epsilon".to_string(), GgufValue::F32(1e-6));
        meta.insert("llama.rope.freq_base".to_string(), GgufValue::F32(10000.0));
        meta.insert("llama.rope.dimension_count".to_string(), GgufValue::U32(64));
        meta.insert("llama.context_length".to_string(), GgufValue::U32(2048));
        meta.insert("tokenizer.ggml.tokens".to_string(), GgufValue::ArrayLen(8192));

        let spec = ModelSpec::from_gguf_metadata(&meta);
        assert_eq!(spec.n_embd, 512);
        assert_eq!(spec.n_layer, 4);
        assert_eq!(spec.n_vocab, 8192);
    }
}
