//! Explicit semantics around the existing dense GPU engine; no new kernels.
use crate::model_descriptor::{
    Architecture, Precision, RopeLayout, TensorBinding, ValidatedModelDescriptor,
};
#[derive(Clone, Debug)]
pub enum Operator {
    RmsNorm {
        weight: String,
        epsilon: f32,
    },
    Linear {
        weight: String,
        input: usize,
        output: usize,
    },
    HeadRmsNorm {
        query_weight: String,
        key_weight: String,
        head_dim: usize,
        epsilon: f32,
    },
    CacheWrite,
    RelativeRopeCausalAttention {
        layout: RopeLayout,
        rope_dimensions: usize,
        rope_base: f32,
        frequency_factors: Option<String>,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    },
    ResidualAdd,
    SiluGateMultiply,
}
#[derive(Clone, Debug)]
pub struct DenseLayerPlan {
    pub index: usize,
    pub operators: Vec<Operator>,
}
#[derive(Clone, Debug)]
/// Layer-local [position, KV head, head dimension]; K is normalized and unrotated.
/// Relative RoPE is applied while computing attention scores. V remains unrotated.
pub struct DenseStatePlan {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub capacity: usize,
    pub precision: Precision,
}
#[derive(Clone, Debug)]
pub struct PrecisionContract {
    pub activations: Precision,
    pub accumulation: Precision,
    pub kv: Precision,
    pub quantized_weights_dequantized_per_access: bool,
}
#[derive(Clone, Debug)]
pub struct DenseExecutionPlan {
    descriptor: ValidatedModelDescriptor,
    layers: Vec<DenseLayerPlan>,
    final_operators: Vec<Operator>,
    state: DenseStatePlan,
    precision: PrecisionContract,
}
impl DenseExecutionPlan {
    pub fn layers(&self) -> &[DenseLayerPlan] {
        &self.layers
    }
    pub fn final_operators(&self) -> &[Operator] {
        &self.final_operators
    }
    pub fn state(&self) -> &DenseStatePlan {
        &self.state
    }
    pub fn precision(&self) -> &PrecisionContract {
        &self.precision
    }

    /// Admission for the unchanged dense kernels, separate from structural planning.
    pub fn check_legacy_adapter(&self) -> Result<(), crate::model_descriptor::AdmissionError> {
        // Admission verifies factor tensor layout; msum preflight validates values
        // and applies the same per-pair factors to native and extended tables.
        Ok(())
    }
    pub fn descriptor(&self) -> &ValidatedModelDescriptor {
        &self.descriptor
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorBinding> {
        self.descriptor.bindings().iter().find(|t| t.name == name)
    }
    pub fn compile(descriptor: ValidatedModelDescriptor) -> Self {
        let d = &descriptor;
        let qwidth = d.heads * d.head_dim;
        let kvwidth = d.kv_heads * d.head_dim;
        let mut layers = Vec::with_capacity(d.layers);
        for index in 0..d.layers {
            let name = |s: &str| format!("blk.{index}.{s}.weight");
            let mut operators = vec![
                Operator::RmsNorm {
                    weight: name("attn_norm"),
                    epsilon: d.rms_epsilon,
                },
                Operator::Linear {
                    weight: name("attn_q"),
                    input: d.hidden,
                    output: qwidth,
                },
                Operator::Linear {
                    weight: name("attn_k"),
                    input: d.hidden,
                    output: kvwidth,
                },
                Operator::Linear {
                    weight: name("attn_v"),
                    input: d.hidden,
                    output: kvwidth,
                },
            ];
            if d.architecture == Architecture::Qwen3 {
                operators.push(Operator::HeadRmsNorm {
                    query_weight: name("attn_q_norm"),
                    key_weight: name("attn_k_norm"),
                    head_dim: d.head_dim,
                    epsilon: d.rms_epsilon,
                });
            }
            operators.extend([
                Operator::CacheWrite,
                Operator::RelativeRopeCausalAttention {
                    layout: if d.architecture == Architecture::Qwen3 {
                        RopeLayout::SplitHalf
                    } else {
                        RopeLayout::AdjacentPairs
                    },
                    rope_dimensions: d.rope_dimensions,
                    rope_base: d.rope_base,
                    frequency_factors: d.rope_frequency_factors.clone(),
                    query_heads: d.heads,
                    kv_heads: d.kv_heads,
                    head_dim: d.head_dim,
                    scale: 1.0 / (d.head_dim as f32).sqrt(),
                },
                Operator::Linear {
                    weight: name("attn_output"),
                    input: qwidth,
                    output: d.hidden,
                },
                Operator::ResidualAdd,
                Operator::RmsNorm {
                    weight: name("ffn_norm"),
                    epsilon: d.rms_epsilon,
                },
                Operator::Linear {
                    weight: name("ffn_gate"),
                    input: d.hidden,
                    output: d.ffn,
                },
                Operator::Linear {
                    weight: name("ffn_up"),
                    input: d.hidden,
                    output: d.ffn,
                },
                Operator::SiluGateMultiply,
                Operator::Linear {
                    weight: name("ffn_down"),
                    input: d.ffn,
                    output: d.hidden,
                },
                Operator::ResidualAdd,
            ]);
            layers.push(DenseLayerPlan { index, operators });
        }
        let final_operators = vec![
            Operator::RmsNorm {
                weight: "output_norm.weight".into(),
                epsilon: d.rms_epsilon,
            },
            Operator::Linear {
                weight: d.output_head.clone(),
                input: d.hidden,
                output: d.vocab,
            },
        ];
        let state = DenseStatePlan {
            layers: d.layers,
            kv_heads: d.kv_heads,
            head_dim: d.head_dim,
            capacity: d.context,
            precision: Precision::F32,
        };
        Self {
            descriptor,
            layers,
            final_operators,
            state,
            precision: PrecisionContract {
                activations: Precision::F32,
                accumulation: Precision::F32,
                kv: Precision::F32,
                quantized_weights_dequantized_per_access: true,
            },
        }
    }
}
