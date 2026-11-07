//! Validated dense acquisition; requires the coordinated B1/B2/B3/msum source set.
use crate::{
    backend::bindless::{metadata::BindlessMetadata, preflight::PreflightResources},
    core::spec::{ModelSpec, NormKind},
    execution::{
        operators::{
            Activation, AffineScale, NormKind as OpNorm, OperatorContract, Pairing, TensorView,
            ValidatedOperator,
        },
        plan::{DenseExecutionPlan, Operator},
    },
    model_descriptor::{AdmissionError, ValidatedModelDescriptor},
    runtime::state::{KvPrecision, StateBudget, StateIdentity, StateKind, StateSpec},
};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use shimmytok::manifest::ModelManifest;
use std::{fs::File, path::Path, sync::Arc};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum ModelAdmissionError {
    #[error(transparent)]
    Descriptor(#[from] AdmissionError),
    #[error("invalid_gguf: {0}")]
    Manifest(#[from] shimmytok::Error),
    #[error("unsupported_tokenizer: {0}")]
    Tokenizer(#[source] shimmytok::Error),
    #[error(transparent)]
    Operator(#[from] super::operators::OperatorError),
    #[error(transparent)]
    State(#[from] crate::runtime::state::StateError),
    #[error(transparent)]
    Rope(#[from] crate::backend::bindless::preflight::RopeFactorError),
    #[error("model_io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{code}: {detail}")]
    Configuration { code: &'static str, detail: String },
    #[error("{code}: {detail}")]
    Device { code: &'static str, detail: String },
}
impl ModelAdmissionError {
    pub fn code(&self) -> &str {
        match self {
            Self::Descriptor(e) => e.code,
            Self::Manifest(_) => "invalid_gguf",
            Self::Tokenizer(_) => "unsupported_tokenizer",
            Self::Operator(_) => "unsupported_operator",
            Self::State(_) => "invalid_state_plan",
            Self::Rope(_) => "invalid_rope_factors",
            Self::Io(_) => "model_io",
            Self::Configuration { code, .. } | Self::Device { code, .. } => code,
        }
    }
    pub fn client_error(&self) -> bool {
        !matches!(self, Self::Io(_) | Self::Device { .. })
    }
}
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("{code}: {detail}")]
    Request { code: &'static str, detail: String },
    #[error(transparent)]
    State(#[from] crate::runtime::state::StateError),
    #[error("gpu_execution: {0}")]
    Gpu(String),
    #[error("{code}: {detail}")]
    Scheduler { code: &'static str, detail: String },
}
fn config(code: &'static str, detail: impl ToString) -> ModelAdmissionError {
    ModelAdmissionError::Configuration {
        code,
        detail: detail.to_string(),
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelInspection {
    pub architecture: Option<String>,
    pub structural_support: bool,
    pub executor_available: bool,
    pub execution_configuration: &'static str,
    pub native_context: Option<usize>,
    pub reason_code: Option<String>,
    pub reason: Option<String>,
    pub numerical_certification: &'static str,
    pub platform_certification: &'static str,
    #[serde(skip_serializing)]
    pub chat_template: Option<String>,
}
pub fn inspect_model(path: &Path) -> Result<ModelInspection, ModelAdmissionError> {
    let manifest = shimmytok::manifest::load_manifest(path)?;
    let architecture = match manifest.header().values.get("general.architecture") {
        Some(shimmytok::gguf::GGUFValue::String(s)) => Some(s.clone()),
        _ => None,
    };
    let chat_template = match manifest.header().values.get("tokenizer.chat_template") {
        Some(shimmytok::gguf::GGUFValue::String(s)) => Some(s.clone()),
        _ => None,
    };
    let mut report = ModelInspection {
        architecture,
        chat_template,
        structural_support: false,
        executor_available: cfg!(feature = "isf"),
        execution_configuration: "dense.f32.reference",
        native_context: None,
        reason_code: None,
        reason: None,
        numerical_certification: "NOT_RUN",
        platform_certification: "NOT_RUN",
    };
    let checked = (|| -> Result<(), ModelAdmissionError> {
        let descriptor = ValidatedModelDescriptor::from_manifest(&manifest)?;
        report.native_context = Some(descriptor.context());
        let plan = DenseExecutionPlan::compile(descriptor);
        let _ = checked_operators(&plan, &manifest)?;
        Ok(())
    })();
    match checked {
        Ok(()) => report.structural_support = true,
        Err(e) => {
            report.reason_code = Some(e.code().to_string());
            report.reason = Some(e.to_string());
        }
    }
    Ok(report)
}
pub struct ValidatedDenseModel {
    manifest: Arc<ModelManifest>,
    plan: DenseExecutionPlan,
    operators: Vec<ValidatedOperator>,
    spec: ModelSpec,
    identity: StateIdentity,
    state: StateSpec,
}
pub struct PreparedDenseModel {
    pub(crate) tokenizer: Arc<shimmytok::Tokenizer>,
    pub(crate) bytes: Mmap,
    pub(crate) validated: Arc<ValidatedDenseModel>,
    pub(crate) metadata: BindlessMetadata,
    pub(crate) rope_tables: (Vec<f32>, Vec<f32>),
}
impl ValidatedDenseModel {
    pub fn plan(&self) -> &DenseExecutionPlan {
        &self.plan
    }
    pub fn identity(&self) -> &StateIdentity {
        &self.identity
    }
    pub fn state_spec(&self) -> &StateSpec {
        &self.state
    }
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
    pub fn check_spec(&self, spec: &ModelSpec) -> Result<(), ModelAdmissionError> {
        if spec != &self.spec {
            return Err(config(
                "model_plan_mismatch",
                "caller ModelSpec differs from validated adapter specification",
            ));
        }
        Ok(())
    }
    pub fn check_device(&self, limits: &wgpu::Limits) -> Result<StateBudget, ModelAdmissionError> {
        let chunk = crate::backend::bindless::loader::compute_chunk_plan(
            self.manifest.file_len(),
            u64::from(limits.max_storage_buffer_binding_size),
        );
        if chunk.effective_chunk > limits.max_buffer_size {
            return Err(ModelAdmissionError::Device {
                code: "device_limit",
                detail: "blob chunk exceeds buffer limit".into(),
            });
        }
        let span = |start: u64, end: u64| -> Result<(), ModelAdmissionError> {
            if start >= end || end > self.manifest.file_len() {
                return Err(config(
                    "invalid_tensor_range",
                    "empty or out-of-file binding window",
                ));
            }
            crate::backend::bindless::loader::BlobWindow::for_range(
                (start / 4) as u32,
                ((end - 1) / 4) as u32,
                (chunk.effective_chunk / 4) as u32,
                chunk.num_chunks,
            )
            .map_err(|detail| ModelAdmissionError::Device {
                code: "device_binding_window",
                detail,
            })?;
            Ok(())
        };
        for layer in 0..self.spec.n_layer {
            let prefix = format!("blk.{layer}.");
            let tensors: Vec<_> = self
                .plan
                .descriptor()
                .bindings()
                .iter()
                .filter(|t| t.name.starts_with(&prefix))
                .collect();
            let start = tensors
                .iter()
                .map(|t| t.offset)
                .min()
                .ok_or_else(|| config("missing_layer", "empty planned layer"))?
                .saturating_sub(4);
            let end = tensors
                .iter()
                .map(|t| t.offset + t.bytes)
                .max()
                .expect("nonempty layer");
            span(start, end)?;
        }
        for name in ["token_embd.weight", "output.weight"] {
            if let Some(t) = self.manifest.tensor(name) {
                span(t.absolute_offset(), t.absolute_offset() + t.byte_len())?;
            }
        }
        let budget = self.state.budget(
            u64::from(limits.max_storage_buffer_binding_size),
            4 * 1024 * 1024 * 1024,
        )?;
        let scratch = (self.spec.temp_buffer_size as u64)
            .checked_mul(4)
            .ok_or_else(|| config("buffer_overflow", "scratch"))?;
        if scratch > limits.max_buffer_size {
            return Err(ModelAdmissionError::Device {
                code: "device_limit",
                detail: "scratch buffer exceeds device limit".into(),
            });
        }
        if limits.max_storage_buffers_per_shader_stage < 14
            || limits.max_compute_invocations_per_workgroup < 256
        {
            return Err(ModelAdmissionError::Device {
                code: "device_limit",
                detail: "dense kernel binding/workgroup requirements unavailable".into(),
            });
        }
        Ok(budget)
    }
    pub fn check_forward(
        &self,
        spec: &ModelSpec,
        rows: usize,
        start: usize,
        end: usize,
    ) -> Result<(), ModelAdmissionError> {
        self.check_spec(spec)?;
        if rows == 0 || start.checked_add(rows) != Some(end) || end > self.state.capacity {
            return Err(config(
                "context_length_exceeded",
                "forward extent does not fit initialized dense state capacity",
            ));
        }
        for node in &self.operators {
            node.encode(|_, _| Ok(()))?;
        }
        Ok(())
    }
}
fn checked_operators(
    plan: &DenseExecutionPlan,
    m: &ModelManifest,
) -> Result<Vec<ValidatedOperator>, ModelAdmissionError> {
    let d = plan.descriptor();
    let mut nodes = Vec::new();
    let activation = |name: &str, dims: &[usize]| TensorView::activation(name, dims);
    let weight = |name: &str| -> Result<TensorView, ModelAdmissionError> {
        let t = m
            .tensor(name)
            .ok_or_else(|| config("missing_tensor", name))?;
        Ok(TensorView::weight(t, m.file_len())?)
    };
    for op in plan
        .layers()
        .iter()
        .flat_map(|l| l.operators.iter())
        .chain(plan.final_operators().iter())
    {
        match op {
            Operator::RmsNorm { weight: w, epsilon } => {
                let x = activation("activation", &[1, d.hidden])?;
                let y = activation("normalized", &[1, d.hidden])?;
                nodes.push(
                    OperatorContract::Norm {
                        kind: OpNorm::Rms,
                        epsilon: *epsilon,
                        affine: AffineScale::Weight,
                        bias: false,
                    }
                    .validate_legacy(&[x, weight(w)?], &y)?,
                );
            }
            Operator::Linear {
                weight: w,
                input,
                output,
            } => {
                let x = activation("activation", &[1, *input])?;
                let y = activation("projected", &[1, *output])?;
                nodes.push(
                    OperatorContract::Linear { bias: false }
                        .validate_legacy(&[x, weight(w)?], &y)?,
                );
            }
            Operator::HeadRmsNorm {
                query_weight,
                key_weight,
                head_dim,
                epsilon,
            } => {
                for (name, heads) in [(query_weight, d.heads), (key_weight, d.kv_heads)] {
                    let x = activation("head activation", &[1, heads, *head_dim])?;
                    let y = x.clone();
                    nodes.push(
                        OperatorContract::HeadNorm {
                            heads,
                            head_dim: *head_dim,
                            epsilon: *epsilon,
                        }
                        .validate_legacy(&[x, weight(name)?], &y)?,
                    );
                }
            }
            Operator::RelativeRopeCausalAttention {
                layout,
                rope_dimensions,
                rope_base,
                frequency_factors,
                query_heads,
                kv_heads,
                head_dim,
                scale,
            } => {
                if let Some(name) = frequency_factors {
                    let _ = weight(name)?;
                }
                let q = activation("Q", &[1, *query_heads, *head_dim])?;
                let k = activation("K", &[1, *kv_heads, *head_dim])?;
                let v = k.clone();
                let out = q.clone();
                nodes.push(
                    OperatorContract::RelativeRopeAttention {
                        query_heads: *query_heads,
                        kv_heads: *kv_heads,
                        head_dim: *head_dim,
                        rope_dim: *rope_dimensions,
                        base: *rope_base,
                        pairing: match layout {
                            crate::model_descriptor::RopeLayout::AdjacentPairs => Pairing::Adjacent,
                            crate::model_descriptor::RopeLayout::SplitHalf => Pairing::SplitHalf,
                        },
                        scale: *scale,
                        softcap: None,
                        query_start: 0,
                    }
                    .validate_legacy(&[q, k, v], &out)?,
                );
            }
            Operator::CacheWrite => {
                let x = activation("KV", &[1, d.kv_heads, d.head_dim])?;
                nodes.push(
                    OperatorContract::CacheWrite {
                        start: 0,
                        count: 1,
                        capacity: d.context,
                    }
                    .validate_legacy(&[x.clone(), x.clone()], &x)?,
                );
            }
            Operator::ResidualAdd => {
                let x = activation("residual", &[1, d.hidden])?;
                nodes.push(
                    OperatorContract::ResidualAdd.validate_legacy(&[x.clone(), x.clone()], &x)?,
                );
            }
            Operator::SiluGateMultiply => {
                let x = activation("FFN", &[1, d.ffn])?;
                nodes.push(
                    OperatorContract::GateMultiply {
                        activation: Activation::Silu,
                    }
                    .validate_legacy(&[x.clone(), x.clone()], &x)?,
                );
            }
        }
    }
    Ok(nodes)
}
fn context_limit(native: usize, requested: Option<usize>) -> Result<usize, ModelAdmissionError> {
    let context = requested.unwrap_or(native.min(8192));
    if context == 0 || context > native {
        return Err(config("unsupported_context","context must be positive and at most declared native capacity; extension is not certified"));
    }
    Ok(context)
}
pub fn configured_context() -> Result<Option<usize>, ModelAdmissionError> {
    match std::env::var("SHIMMY_MAX_CTX") {
        Ok(value) => value.parse::<usize>().map(Some).map_err(|_| {
            config(
                "invalid_context",
                "SHIMMY_MAX_CTX must be an unsigned integer",
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(config("invalid_context", "SHIMMY_MAX_CTX is not UTF-8")),
    }
}
pub fn require_diagnostic_execution() -> Result<(), ModelAdmissionError> {
    if std::env::var("AIRFRAME_UNVALIDATED_DIAGNOSTICS").is_ok_and(|v| v == "1") {
        Ok(())
    } else {
        Err(config("unsupported_diagnostic_executor","single-layer helpers are unvalidated diagnostics; use the validated full-model executor, or explicitly set AIRFRAME_UNVALIDATED_DIAGNOSTICS=1 for private diagnostics"))
    }
}
pub fn validate_kernel_configuration() -> Result<(), ModelAdmissionError> {
    for flag in [
        "AIRFRAME_EXPERIMENTAL_STAGED_FFN",
        "AIRFRAME_EXPERIMENTAL_VECTORIZED_FFN",
        "SHIMMY_DISABLE_OUTPUT_NORM",
        "SHIMMY_ROUTE_V2_LAYER_PARAMS",
        "AIRFRAME_PINGPONG_ACTIVATION",
        "AIRFRAME_EXPERIMENTAL_COOPERATIVE_FFN_DOWN",
        "AIRFRAME_EXPERIMENTAL_COOPERATIVE_QKV",
        "AIRFRAME_EXPERIMENTAL_COOPERATIVE_ATTN_PROJ",
        "AIRFRAME_EXPERIMENTAL_KV_ZERO_UPLOAD",
    ] {
        if std::env::var(flag).is_ok_and(|s| s == "1" || s.eq_ignore_ascii_case("true")) {
            return Err(config(
                "unsupported_kernel_configuration",
                format!("{flag} is outside the correctness candidate"),
            ));
        }
    }
    if std::env::var("SHIMMY_KV_QUANT")
        .or_else(|_| std::env::var("AIRFRAME_KV_QUANT"))
        .is_ok_and(|s| s != "f32" && !s.is_empty())
    {
        return Err(config(
            "unsupported_state_precision",
            "INT4 requires a separately validated operator/state plan",
        ));
    }
    if let Ok(value) = std::env::var("SHIMMY_ROPE_SCALE") {
        if value.parse::<f32>().ok() != Some(1.0) {
            return Err(config(
                "unsupported_position_scaling",
                "runtime RoPE extension/scaling is not certified",
            ));
        }
    }
    Ok(())
}
impl PreparedDenseModel {
    pub fn load(path: &Path, context: Option<usize>) -> Result<Self, ModelAdmissionError> {
        validate_kernel_configuration()?;
        #[cfg(not(feature = "isf"))]
        return Err(config(
            "unsupported_executor",
            "validated dense generation requires the isf executor",
        ));
        #[cfg(feature = "isf")]
        {
            let file = File::open(path)?;
            let bytes = unsafe { Mmap::map(&file)? };
            let manifest = Arc::new(shimmytok::manifest::read_manifest(
                std::io::Cursor::new(&bytes[..]),
                bytes.len() as u64,
                Default::default(),
            )?);
            if bytes.len() as u64 > u64::from(u32::MAX) {
                return Err(config(
                    "unsupported_address_width",
                    "legacy embedding offsets exceed u32 address space",
                ));
            }
            let descriptor = ValidatedModelDescriptor::from_manifest(&manifest)?;
            let tokenizer = Arc::new(
                shimmytok::Tokenizer::from_bytes(&bytes).map_err(ModelAdmissionError::Tokenizer)?,
            );
            for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
                // GPT-2/Qwen vocabularies may intentionally omit BOS. Never
                // invent one; require it only when insertion is requested.
                if key == "tokenizer.ggml.bos_token_id"
                    && !manifest.header().values.contains_key(key)
                {
                    let add_bos = match manifest.header().values.get("tokenizer.ggml.add_bos_token")
                    {
                        Some(shimmytok::gguf::GGUFValue::Bool(value)) => *value,
                        None => {
                            matches!(manifest.header().values.get("tokenizer.ggml.model"),Some(shimmytok::gguf::GGUFValue::String(value)) if value=="llama"||value=="spm")
                        }
                        _ => {
                            return Err(config(
                                "invalid_special_token_metadata",
                                "tokenizer.ggml.add_bos_token",
                            ))
                        }
                    };
                    if !add_bos {
                        continue;
                    }
                }
                match manifest.header().values.get(key) {
                    Some(shimmytok::gguf::GGUFValue::U32(id))
                        if (*id as usize) < descriptor.vocabulary() => {}
                    _ => return Err(config("invalid_special_token_metadata", key)),
                }
            }
            let capacity = context_limit(descriptor.context(), context)?;
            let plan = DenseExecutionPlan::compile(descriptor);
            let operators = checked_operators(&plan, &manifest)?;
            let metadata = BindlessMetadata::from_manifest(&manifest)?;
            let d = plan.descriptor();
            for (name, n) in [
                ("hidden", d.hidden),
                ("heads", d.heads),
                ("kv_heads", d.kv_heads),
                ("head_dim", d.head_dim),
                ("ffn", d.ffn),
                ("layers", d.layers),
                ("vocabulary", d.vocab),
                ("context", capacity),
            ] {
                if n > u32::MAX as usize {
                    return Err(config("unsupported_index_width", name));
                }
            }
            let mut spec = metadata.to_model_spec();
            if spec.norm_kind != NormKind::Rms
                || spec.post_norm_enabled
                || spec.attn_logit_softcap != 0.0
                || spec.final_logit_softcap != 0.0
            {
                return Err(config(
                    "unsupported_operator_semantics",
                    "legacy adapter disagrees with dense RMSNorm/uncapped plan",
                ));
            }
            if spec.n_embd != d.hidden
                || spec.n_head != d.heads
                || spec.n_head_kv != d.kv_heads
                || spec.n_layer != d.layers
                || spec.ff_dim != d.ffn
                || spec.head_dim != d.head_dim
                || spec.n_vocab != d.vocab
                || spec.rms_eps != d.rms_epsilon
                || spec.rope_dim != d.rope_dimensions
                || spec.rope_base != d.rope_base
            {
                return Err(config(
                    "model_plan_mismatch",
                    "legacy dimensions/position/norm metadata differ from checked descriptor",
                ));
            }
            for (name, n) in [
                ("Q width", d.heads.checked_mul(d.head_dim)),
                ("KV width", d.kv_heads.checked_mul(d.head_dim)),
                (
                    "scratch width",
                    d.ffn.checked_mul(2).and_then(|n| n.checked_add(d.hidden)),
                ),
            ] {
                if n.is_none_or(|v| v > u32::MAX as usize) {
                    return Err(config("unsupported_index_width", name));
                }
            }
            spec.n_ctx = capacity;
            spec.rope_scale = 1.0;
            spec = spec.compute_derived();
            let state = StateSpec {
                kind: StateKind::DenseKv,
                layers: d.layers,
                kv_heads: d.kv_heads,
                head_dim: d.head_dim,
                capacity,
                precision: KvPrecision::F32,
            };
            state.budget(u64::MAX, 4 * 1024 * 1024 * 1024)?;
            let rope_tables = PreflightResources::prepare_rope_tables(&bytes, &metadata, &spec)?;
            let model_sha256 = format!("{:x}", Sha256::digest(&bytes[..]));
            let plan_sha256 = format!(
                "{:x}",
                Sha256::digest(format!("dense.f32.v1:{plan:?}:{spec:?}").as_bytes())
            );
            let identity = StateIdentity {
                model_sha256,
                plan_sha256,
            };
            let validated = Arc::new(ValidatedDenseModel {
                manifest,
                plan,
                operators,
                spec,
                identity,
                state,
            });
            Ok(Self {
                tokenizer,
                bytes,
                validated,
                metadata,
                rope_tables,
            })
        }
    }
    pub fn validated(&self) -> &Arc<ValidatedDenseModel> {
        &self.validated
    }
}
