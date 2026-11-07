//! Checked dense foundation. Construction is CPU-only and does not certify execution.
use shimmytok::gguf::GGUFValue as V;
use shimmytok::manifest::{ModelManifest, TensorDescriptor};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Qwen3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    AdjacentPairs,
    SplitHalf,
}
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{code}: {subject}: {detail}")]
pub struct AdmissionError {
    pub code: &'static str,
    pub subject: String,
    pub detail: String,
}
fn error(code: &'static str, subject: &str, detail: impl ToString) -> AdmissionError {
    AdmissionError {
        code,
        subject: subject.into(),
        detail: detail.to_string(),
    }
}
pub(crate) type Result<T> = std::result::Result<T, AdmissionError>;
#[derive(Clone, Debug)]
pub struct TensorBinding {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub bytes: u64,
}
#[derive(Clone, Debug)]
pub struct ValidatedModelDescriptor {
    pub(crate) architecture: Architecture,
    pub(crate) context: usize,
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) ffn: usize,
    pub(crate) layers: usize,
    pub(crate) vocab: usize,
    pub(crate) rms_epsilon: f32,
    pub(crate) rope_base: f32,
    pub(crate) rope_dimensions: usize,
    pub(crate) rope_frequency_factors: Option<String>,
    pub(crate) bindings: Vec<TensorBinding>,
    pub(crate) output_head: String,
}
impl ValidatedModelDescriptor {
    pub fn architecture(&self) -> Architecture {
        self.architecture
    }
    pub fn context(&self) -> usize {
        self.context
    }
    pub fn layer_count(&self) -> usize {
        self.layers
    }
    pub fn vocabulary(&self) -> usize {
        self.vocab
    }
    pub fn bindings(&self) -> &[TensorBinding] {
        &self.bindings
    }
    pub fn from_manifest(m: &ModelManifest) -> Result<Self> {
        let metadata = &m.header().values;
        let architecture = match metadata.get("general.architecture") {
            Some(V::String(s)) if s == "llama" => Architecture::Llama,
            Some(V::String(s)) if s == "qwen3" => Architecture::Qwen3,
            Some(V::String(s)) => {
                return Err(error(
                    "unsupported_architecture",
                    s,
                    "dense foundation supports llama and qwen3 only",
                ))
            }
            _ => {
                return Err(error(
                    "missing_metadata",
                    "general.architecture",
                    "required string",
                ))
            }
        };
        let prefix = match architecture {
            Architecture::Llama => "llama",
            Architecture::Qwen3 => "qwen3",
        };
        let integer = |key: &str| -> Result<usize> {
            match metadata.get(key) {
                Some(V::U32(x)) => {
                    usize::try_from(*x).map_err(|_| error("overflow", key, "platform"))
                }
                Some(V::U64(x)) => {
                    usize::try_from(*x).map_err(|_| error("overflow", key, "platform"))
                }
                Some(_) => Err(error(
                    "invalid_metadata_type",
                    key,
                    "required unsigned integer",
                )),
                None => Err(error("missing_metadata", key, "required unsigned integer")),
            }
        };
        let required = |suffix: &str| -> Result<usize> {
            let key = format!("{prefix}.{suffix}");
            let x = integer(&key)?;
            if x == 0 {
                Err(error("invalid_dimension", &key, "must be positive"))
            } else {
                Ok(x)
            }
        };
        let float = |suffix: &str| -> Result<f32> {
            let key = format!("{prefix}.{suffix}");
            match metadata.get(&key) {
                Some(V::F32(x)) if x.is_finite() && *x > 0.0 => Ok(*x),
                Some(_) => Err(error(
                    "invalid_metadata",
                    &key,
                    "required positive finite F32",
                )),
                None => Err(error(
                    "missing_metadata",
                    &key,
                    "required positive finite F32",
                )),
            }
        };
        for (key, value) in metadata {
            if key.ends_with(".expert_count")
                || key.ends_with(".expert_used_count")
                || key.contains(".ssm.")
                || key.contains(".recurrent.")
                || key.contains("lora_rank")
                || key.ends_with(".attention.sliding_window")
            {
                let zero = matches!(value, V::U32(0) | V::U64(0));
                if !zero {
                    return Err(error(
                        "unsupported_execution_requirement",
                        key,
                        "requires a separate executor/state plan",
                    ));
                }
            }
        }
        let hidden = required("embedding_length")?;
        let heads = required("attention.head_count")?;
        let kv_heads = required("attention.head_count_kv")?;
        if heads % kv_heads != 0 {
            return Err(error(
                "invalid_gqa",
                "attention.head_count",
                "Q heads must be divisible by KV heads",
            ));
        }
        let head_dim = match architecture {
            Architecture::Qwen3 => required("attention.key_length")?,
            Architecture::Llama => {
                let k = format!("{prefix}.attention.key_length");
                if metadata.contains_key(&k) {
                    required("attention.key_length")?
                } else {
                    if hidden % heads != 0 {
                        return Err(error(
                            "invalid_head_dimension",
                            &k,
                            "hidden width must divide head count",
                        ));
                    }
                    hidden / heads
                }
            }
        };
        let vkey = format!("{prefix}.attention.value_length");
        if metadata.contains_key(&vkey) && required("attention.value_length")? != head_dim {
            return Err(error(
                "unsupported_value_dimension",
                &vkey,
                "legacy dense adapter requires equal key/value dimensions",
            ));
        }
        let rope_dimensions = if metadata.contains_key(&format!("{prefix}.rope.dimension_count")) {
            required("rope.dimension_count")?
        } else {
            head_dim
        };
        if rope_dimensions > head_dim || rope_dimensions % 2 != 0 {
            return Err(error(
                "invalid_rope",
                "rope.dimension_count",
                "must be even and at most head dimension",
            ));
        }
        let layers = required("block_count")?;
        let context = required("context_length")?;
        let ffn = required("feed_forward_length")?;
        let rms_epsilon = float("attention.layer_norm_rms_epsilon")?;
        let rope_base = float("rope.freq_base")?;
        let embedding = m
            .tensor("token_embd.weight")
            .ok_or_else(|| error("missing_tensor", "token_embd.weight", "required"))?;
        if embedding.dimensions().len() != 2 || embedding.dimensions()[0] != hidden as u64 {
            return Err(error(
                "invalid_tensor_shape",
                "token_embd.weight",
                "expected [hidden,vocabulary]",
            ));
        }
        let vocab = usize::try_from(embedding.dimensions()[1])
            .map_err(|_| error("overflow", "vocabulary", "platform"))?;
        if let Some(V::Array(tokens)) = metadata.get("tokenizer.ggml.tokens") {
            if tokens.len() != vocab {
                return Err(error(
                    "vocabulary_mismatch",
                    "tokenizer.ggml.tokens",
                    "token count differs from embedding rows",
                ));
            }
        }
        let mut bindings = Vec::new();
        let mut bind = |name: &str, shape: &[usize], norm: bool| -> Result<()> {
            let t = m
                .tensor(name)
                .ok_or_else(|| error("missing_tensor", name, "required by dense graph"))?;
            validate_tensor(t, shape, norm)?;
            bindings.push(TensorBinding {
                name: name.into(),
                dimensions: t.dimensions().to_vec(),
                ggml_type: t.ggml_type(),
                offset: t.absolute_offset(),
                bytes: t.byte_len(),
            });
            Ok(())
        };
        bind("token_embd.weight", &[hidden, vocab], false)?;
        bind("output_norm.weight", &[hidden], true)?;
        let output_head = if m.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        }
        .to_string();
        bind(&output_head, &[hidden, vocab], false)?;
        let rope_frequency_factors = if m.tensor("rope_freqs.weight").is_some() {
            bind("rope_freqs.weight", &[rope_dimensions / 2], true)?;
            Some("rope_freqs.weight".to_string())
        } else {
            None
        };
        let qwidth = heads
            .checked_mul(head_dim)
            .ok_or_else(|| error("overflow", "Q width", "heads * head_dim"))?;
        let kvwidth = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| error("overflow", "KV width", "heads * head_dim"))?;
        for layer in 0..layers {
            for (suffix, shape, norm) in [
                ("attn_norm.weight", vec![hidden], true),
                ("attn_q.weight", vec![hidden, qwidth], false),
                ("attn_k.weight", vec![hidden, kvwidth], false),
                ("attn_v.weight", vec![hidden, kvwidth], false),
                ("attn_output.weight", vec![qwidth, hidden], false),
                ("ffn_norm.weight", vec![hidden], true),
                ("ffn_gate.weight", vec![hidden, ffn], false),
                ("ffn_up.weight", vec![hidden, ffn], false),
                ("ffn_down.weight", vec![ffn, hidden], false),
            ] {
                bind(&format!("blk.{layer}.{suffix}"), &shape, norm)?;
            }
            if architecture == Architecture::Qwen3 {
                for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                    bind(&format!("blk.{layer}.{suffix}"), &[head_dim], true)?;
                }
            }
        }
        for (key, value) in metadata {
            if key.contains(".rope.scaling.") && !matches!(value,V::String(s) if s=="none") {
                return Err(error(
                    "unsupported_position_semantics",
                    key,
                    "scaling requires an explicit position plan",
                ));
            }
            if (key.contains("softcapping") || key.ends_with(".attention.scale"))
                && !matches!(value,V::F32(x) if *x==0.0)
            {
                return Err(error(
                    "unsupported_attention_semantics",
                    key,
                    "requires an explicit score transform",
                ));
            }
        }
        // Refuse graph-affecting tensors this first adapter does not represent.
        for t in m.tensors() {
            if !bindings.iter().any(|b| b.name == t.name()) {
                return Err(error(
                    "unplanned_tensor",
                    t.name(),
                    "all tensors must have an explicit graph role",
                ));
            }
            if t.name().ends_with(".bias")
                || t.name().contains("_exps.")
                || t.name().contains("attn_qkv")
                || t.name().contains("post_attention_norm")
                || t.name().contains("post_ffw_norm")
            {
                return Err(error(
                    "unsupported_tensor_semantics",
                    t.name(),
                    "requires an explicit graph builder",
                ));
            }
        }
        Ok(Self {
            architecture,
            context,
            hidden,
            heads,
            kv_heads,
            head_dim,
            ffn,
            layers,
            vocab,
            rms_epsilon,
            rope_base,
            rope_dimensions,
            rope_frequency_factors,
            bindings,
            output_head,
        })
    }
}
fn validate_tensor(t: &TensorDescriptor, shape: &[usize], norm: bool) -> Result<()> {
    if t.dimensions()
        .iter()
        .copied()
        .ne(shape.iter().map(|x| *x as u64))
    {
        return Err(error(
            "invalid_tensor_shape",
            t.name(),
            format!("expected {shape:?}, got {:?}", t.dimensions()),
        ));
    }
    if norm && t.ggml_type() != 0 {
        return Err(error(
            "unsupported_norm_precision",
            t.name(),
            "existing adapter expects F32 norm weights",
        ));
    }
    if !matches!(t.ggml_type(), 0 | 1 | 2 | 6 | 8 | 12 | 13 | 14) {
        return Err(error(
            "unsupported_operator_quantization",
            t.name(),
            t.ggml_type(),
        ));
    }
    Ok(())
}
