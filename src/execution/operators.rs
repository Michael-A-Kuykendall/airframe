//! CPU validation boundary for the existing dense kernels. No alternate math backend.
use shimmytok::manifest::TensorDescriptor;
use thiserror::Error;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Layer,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineScale {
    Weight,
    OnePlusWeight,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pairing {
    Adjacent,
    SplitHalf,
}
#[derive(Clone, Debug, PartialEq, Error)]
pub enum OperatorError {
    #[error("operator_shape: {0}")]
    Shape(String),
    #[error("operator_bounds: {0}")]
    Bounds(String),
    #[error("operator_precision: {0}")]
    Precision(String),
    #[error("unsupported_operator: {0}")]
    Unsupported(String),
    #[error("operator_parameter: {0}")]
    Parameter(String),
}
type Result<T> = std::result::Result<T, OperatorError>;
#[derive(Clone, Debug)]
pub struct TensorView {
    name: String,
    dimensions: Vec<usize>,
    ggml_type: u32,
    offset: u64,
    bytes: u64,
}
impl TensorView {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn activation(name: &str, dimensions: &[usize]) -> Result<Self> {
        if dimensions.is_empty() || dimensions.contains(&0) {
            return Err(OperatorError::Shape(name.into()));
        }
        let count = dimensions
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or_else(|| OperatorError::Bounds(name.into()))?;
        let bytes = count
            .checked_mul(4)
            .and_then(|x| u64::try_from(x).ok())
            .ok_or_else(|| OperatorError::Bounds(name.into()))?;
        Ok(Self {
            name: name.into(),
            dimensions: dimensions.to_vec(),
            ggml_type: 0,
            offset: 0,
            bytes,
        })
    }
    pub fn weight(t: &TensorDescriptor, file_bytes: u64) -> Result<Self> {
        let end = t
            .absolute_offset()
            .checked_add(t.byte_len())
            .ok_or_else(|| OperatorError::Bounds(t.name().into()))?;
        if end > file_bytes {
            return Err(OperatorError::Bounds(t.name().into()));
        }
        let dimensions = t
            .dimensions()
            .iter()
            .map(|n| usize::try_from(*n).map_err(|_| OperatorError::Bounds(t.name().into())))
            .collect::<Result<Vec<_>>>()?;
        if dimensions.is_empty() || dimensions.contains(&0) {
            return Err(OperatorError::Shape(t.name().into()));
        }
        Ok(Self {
            name: t.name().into(),
            dimensions,
            ggml_type: t.ggml_type(),
            offset: t.absolute_offset(),
            bytes: t.byte_len(),
        })
    }
    fn f32(&self) -> Result<()> {
        if self.ggml_type == 0 {
            Ok(())
        } else {
            Err(OperatorError::Precision(self.name.clone()))
        }
    }
}
#[derive(Clone, Debug)]
pub struct PrecisionPolicy {
    pub activations: Precision,
    pub accumulation: Precision,
    pub state: Precision,
}
impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            activations: Precision::F32,
            accumulation: Precision::F32,
            state: Precision::F32,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateEffects {
    None,
    ReadDenseKv { positions: usize },
    WriteDenseKv { start: usize, count: usize },
}
#[derive(Clone, Debug)]
pub enum OperatorContract {
    Norm {
        kind: NormKind,
        epsilon: f32,
        affine: AffineScale,
        bias: bool,
    },
    Linear {
        bias: bool,
    },
    HeadNorm {
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    },
    RelativeRopeAttention {
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        base: f32,
        pairing: Pairing,
        scale: f32,
        softcap: Option<f32>,
        query_start: usize,
    },
    GateMultiply {
        activation: Activation,
    },
    ResidualAdd,
    CacheWrite {
        start: usize,
        count: usize,
        capacity: usize,
    },
    Unsupported {
        operation: String,
    },
}
#[derive(Clone, Debug)]
pub struct OutputContract {
    pub dimensions: Vec<usize>,
    pub precision: Precision,
    pub effects: StateEffects,
}
fn positive(name: &str, x: f32) -> Result<()> {
    if x.is_finite() && x > 0.0 {
        Ok(())
    } else {
        Err(OperatorError::Parameter(name.into()))
    }
}
fn arity(inputs: &[TensorView], n: usize) -> Result<()> {
    if inputs.len() == n {
        Ok(())
    } else {
        Err(OperatorError::Shape(format!(
            "expected {n} inputs, got {}",
            inputs.len()
        )))
    }
}
fn shape(t: &TensorView, expected: &[usize]) -> Result<()> {
    if t.dimensions == expected {
        Ok(())
    } else {
        Err(OperatorError::Shape(format!(
            "{} expected {expected:?}, got {:?}",
            t.name, t.dimensions
        )))
    }
}
impl OperatorContract {
    pub fn infer_shapes(&self, inputs: &[TensorView]) -> Result<OutputContract> {
        let mut effects = StateEffects::None;
        let dimensions = match self {
            Self::Norm { epsilon, bias, .. } => {
                arity(inputs, 2 + usize::from(*bias))?;
                positive("norm epsilon", *epsilon)?;
                let x = &inputs[0];
                x.f32()?;
                inputs[1].f32()?;
                let width = *x
                    .dimensions
                    .last()
                    .ok_or_else(|| OperatorError::Shape(x.name.clone()))?;
                shape(&inputs[1], &[width])?;
                if *bias {
                    inputs[2].f32()?;
                    shape(&inputs[2], &[width])?;
                }
                x.dimensions.clone()
            }
            Self::Linear { bias } => {
                arity(inputs, 2 + usize::from(*bias))?;
                let x = &inputs[0];
                let w = &inputs[1];
                x.f32()?;
                if x.dimensions.len() != 2 || w.dimensions.len() != 2 {
                    return Err(OperatorError::Shape(
                        "linear expects activation [batch,input] and GGUF weight [input,output]"
                            .into(),
                    ));
                }
                if x.dimensions[1] != w.dimensions[0] {
                    return Err(OperatorError::Shape(
                        "linear inner dimensions differ".into(),
                    ));
                }
                if !matches!(w.ggml_type, 0 | 1 | 2 | 6 | 8 | 12 | 13 | 14) {
                    return Err(OperatorError::Unsupported(format!(
                        "linear GGML type {}",
                        w.ggml_type
                    )));
                }
                if *bias {
                    inputs[2].f32()?;
                    shape(&inputs[2], &[w.dimensions[1]])?;
                }
                vec![x.dimensions[0], w.dimensions[1]]
            }
            Self::HeadNorm {
                heads,
                head_dim,
                epsilon,
            } => {
                arity(inputs, 2)?;
                positive("head norm epsilon", *epsilon)?;
                inputs[0].f32()?;
                inputs[1].f32()?;
                let x = &inputs[0];
                if *heads == 0
                    || *head_dim == 0
                    || x.dimensions.len() != 3
                    || x.dimensions[1..] != [*heads, *head_dim]
                {
                    return Err(OperatorError::Shape(
                        "head norm expects [batch,heads,head_dim]".into(),
                    ));
                }
                shape(&inputs[1], &[*head_dim])?;
                x.dimensions.clone()
            }
            Self::RelativeRopeAttention {
                query_heads,
                kv_heads,
                head_dim,
                rope_dim,
                base,
                scale,
                softcap,
                query_start,
                ..
            } => {
                arity(inputs, 3)?;
                positive("rope base", *base)?;
                positive("attention scale", *scale)?;
                if let Some(cap) = softcap {
                    positive("softcap", *cap)?;
                }
                if *kv_heads == 0
                    || *query_heads == 0
                    || *head_dim == 0
                    || query_heads % kv_heads != 0
                    || *rope_dim == 0
                    || rope_dim % 2 != 0
                    || rope_dim > head_dim
                {
                    return Err(OperatorError::Parameter(
                        "invalid attention geometry".into(),
                    ));
                }
                for t in inputs {
                    t.f32()?;
                }
                let q = &inputs[0];
                let k = &inputs[1];
                let v = &inputs[2];
                if q.dimensions.len() != 3 || k.dimensions.len() != 3 {
                    return Err(OperatorError::Shape(
                        "attention expects rank 3 Q/K/V".into(),
                    ));
                }
                let batch = q.dimensions[0];
                let positions = k.dimensions[0];
                shape(q, &[batch, *query_heads, *head_dim])?;
                shape(k, &[positions, *kv_heads, *head_dim])?;
                shape(v, &k.dimensions)?;
                if query_start.checked_add(batch) != Some(positions) {
                    return Err(OperatorError::Bounds(
                        "causal query extent must equal initialized KV extent".into(),
                    ));
                }
                effects = StateEffects::ReadDenseKv { positions };
                vec![batch, *query_heads, *head_dim]
            }
            Self::GateMultiply { .. } | Self::ResidualAdd => {
                arity(inputs, 2)?;
                inputs[0].f32()?;
                inputs[1].f32()?;
                shape(&inputs[1], &inputs[0].dimensions)?;
                inputs[0].dimensions.clone()
            }
            Self::CacheWrite {
                start,
                count,
                capacity,
            } => {
                arity(inputs, 2)?;
                inputs[0].f32()?;
                inputs[1].f32()?;
                shape(&inputs[1], &inputs[0].dimensions)?;
                if *count == 0
                    || inputs[0].dimensions.len() != 3
                    || inputs[0].dimensions[0] != *count
                    || start.checked_add(*count).is_none_or(|end| end > *capacity)
                {
                    return Err(OperatorError::Bounds("KV write extent".into()));
                }
                effects = StateEffects::WriteDenseKv {
                    start: *start,
                    count: *count,
                };
                inputs[0].dimensions.clone()
            }
            Self::Unsupported { operation } => {
                return Err(OperatorError::Unsupported(operation.clone()))
            }
        };
        Ok(OutputContract {
            dimensions,
            precision: Precision::F32,
            effects,
        })
    }
    pub fn required_features(&self) -> Result<&'static [&'static str]> {
        match self {
            Self::Unsupported { operation } => Err(OperatorError::Unsupported(operation.clone())),
            Self::RelativeRopeAttention { .. } => {
                Ok(&["f32_storage", "unrotated_dense_kv", "relative_rope"])
            }
            Self::CacheWrite { .. } => Ok(&["f32_storage", "dense_kv"]),
            _ => Ok(&["f32_storage"]),
        }
    }
    pub fn validate_legacy(
        &self,
        inputs: &[TensorView],
        output: &TensorView,
    ) -> Result<ValidatedOperator> {
        self.required_features()?;
        let inferred = self.infer_shapes(inputs)?;
        output.f32()?;
        shape(output, &inferred.dimensions)?;
        match self {
            Self::Norm {
                kind: NormKind::Layer,
                ..
            }
            | Self::Norm {
                affine: AffineScale::OnePlusWeight,
                ..
            }
            | Self::GateMultiply {
                activation: Activation::GeluTanh,
            }
            | Self::RelativeRopeAttention {
                softcap: Some(_), ..
            } => return Err(OperatorError::Unsupported(
                "first dense adapter certifies RMSNorm(weight), SiLU and uncapped attention only"
                    .into(),
            )),
            _ => {}
        }
        Ok(ValidatedOperator {
            contract: self.clone(),
            output: inferred,
        })
    }
}
#[derive(Clone, Debug)]
pub struct ValidatedOperator {
    contract: OperatorContract,
    output: OutputContract,
}
impl ValidatedOperator {
    pub fn output(&self) -> &OutputContract {
        &self.output
    }
    pub fn contract(&self) -> &OperatorContract {
        &self.contract
    }
    /// Only a successfully checked contract can reach the caller's existing encoder.
    pub fn encode<T>(
        &self,
        encode: impl FnOnce(&OperatorContract, &OutputContract) -> Result<T>,
    ) -> Result<T> {
        encode(&self.contract, &self.output)
    }
}
