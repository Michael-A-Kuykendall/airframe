use airframe::execution::operators::*;
fn a(name: &str, shape: &[usize]) -> TensorView {
    TensorView::activation(name, shape).unwrap()
}
#[test]
fn operator_contract_rejects_unsafe_shapes_before_encoding() {
    let op = OperatorContract::Linear { bias: false };
    let inputs = [a("x", &[2, 3]), a("w", &[4, 5])];
    let out = a("y", &[2, 5]);
    assert!(op.validate_legacy(&inputs, &out).is_err());
    assert!(TensorView::activation("overflow", &[usize::MAX, 2]).is_err());
    assert!(TensorView::activation("empty", &[1, 0]).is_err());
}
#[test]
fn operator_contract_norm_semantics_do_not_alias() {
    let inputs = [a("x", &[3, 8]), a("weight", &[8])];
    let out = a("y", &[3, 8]);
    let rms = OperatorContract::Norm {
        kind: NormKind::Rms,
        epsilon: 1e-6,
        affine: AffineScale::Weight,
        bias: false,
    };
    assert!(rms.validate_legacy(&inputs, &out).is_ok());
    for (kind, affine) in [
        (NormKind::Layer, AffineScale::Weight),
        (NormKind::Rms, AffineScale::OnePlusWeight),
    ] {
        let op = OperatorContract::Norm {
            kind,
            epsilon: 1e-6,
            affine,
            bias: false,
        };
        assert!(op.infer_shapes(&inputs).is_ok());
        assert!(op.validate_legacy(&inputs, &out).is_err());
    }
    let invalid = OperatorContract::Norm {
        kind: NormKind::Rms,
        epsilon: f32::NAN,
        affine: AffineScale::Weight,
        bias: false,
    };
    assert!(invalid.infer_shapes(&inputs).is_err());
}
#[test]
fn operator_contract_head_strides_and_causal_extent_are_explicit() {
    let head = OperatorContract::HeadNorm {
        heads: 16,
        head_dim: 128,
        epsilon: 1e-6,
    };
    assert!(head
        .infer_shapes(&[a("q", &[3, 16, 128]), a("norm", &[128])])
        .is_ok());
    assert!(head
        .infer_shapes(&[a("q", &[3, 128, 16]), a("norm", &[128])])
        .is_err());
    let op = OperatorContract::RelativeRopeAttention {
        query_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        rope_dim: 128,
        base: 1e6,
        pairing: Pairing::SplitHalf,
        scale: 1.0 / 128.0f32.sqrt(),
        softcap: None,
        query_start: 4,
    };
    let valid = op
        .infer_shapes(&[
            a("q", &[2, 16, 128]),
            a("k", &[6, 8, 128]),
            a("v", &[6, 8, 128]),
        ])
        .unwrap();
    assert_eq!(valid.effects, StateEffects::ReadDenseKv { positions: 6 });
    assert!(op
        .infer_shapes(&[
            a("q", &[2, 16, 128]),
            a("k", &[7, 8, 128]),
            a("v", &[7, 8, 128])
        ])
        .is_err());
}
#[test]
fn operator_contract_gate_activation_independent_of_attention() {
    let inputs = [a("gate", &[2, 3072]), a("up", &[2, 3072])];
    let output = a("ffn", &[2, 3072]);
    assert!(OperatorContract::GateMultiply {
        activation: Activation::Silu
    }
    .validate_legacy(&inputs, &output)
    .is_ok());
    let gelu = OperatorContract::GateMultiply {
        activation: Activation::GeluTanh,
    };
    assert!(gelu.infer_shapes(&inputs).is_ok());
    assert!(gelu.validate_legacy(&inputs, &output).is_err());
}
#[test]
fn operator_contract_checks_write_extent_and_exposes_state_effects() {
    let op = OperatorContract::CacheWrite {
        start: 6,
        count: 2,
        capacity: 8,
    };
    let inputs = [a("k", &[2, 8, 128]), a("v", &[2, 8, 128])];
    assert_eq!(
        op.infer_shapes(&inputs).unwrap().effects,
        StateEffects::WriteDenseKv { start: 6, count: 2 }
    );
    assert!(OperatorContract::CacheWrite {
        start: usize::MAX,
        count: 2,
        capacity: usize::MAX
    }
    .infer_shapes(&inputs)
    .is_err());
}
#[test]
fn operator_contract_only_validated_node_reaches_encoder() {
    let op = OperatorContract::ResidualAdd;
    let inputs = [a("lhs", &[2, 4]), a("rhs", &[2, 4])];
    let output = a("out", &[2, 4]);
    let validated = op.validate_legacy(&inputs, &output).unwrap();
    let mut called = 0;
    validated
        .encode(|_, out| {
            assert_eq!(out.dimensions, [2, 4]);
            called += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(called, 1);
    assert!(OperatorContract::Unsupported {
        operation: "MLA".into()
    }
    .required_features()
    .is_err());
}
