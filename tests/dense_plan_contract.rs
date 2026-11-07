use airframe::{
    execution::plan::{DenseExecutionPlan, Operator},
    model_descriptor::{Architecture, ValidatedModelDescriptor},
};
#[test]
#[ignore = "requires exact local GGUF fixtures"]
fn exact_qwen_and_llama_compile_dense_plans() {
    let root = std::env::var("SHIMMY_CERT_MODEL_DIR").expect("fixtures");
    for (name, family) in [
        ("Qwen3-0.6B-Q4_K_M.gguf", Architecture::Qwen3),
        ("llama3.2-3b-instruct-dde5aa3f.gguf", Architecture::Llama),
    ] {
        let m = shimmytok::manifest::load_manifest(std::path::Path::new(&root).join(name)).unwrap();
        let descriptor = ValidatedModelDescriptor::from_manifest(&m).unwrap();
        assert_eq!(descriptor.architecture(), family);
        assert_eq!(descriptor.layer_count(), 28);
        let p = DenseExecutionPlan::compile(descriptor);
        // Coordinated adapter includes airframe-msum factor preparation.
        assert!(p.check_legacy_adapter().is_ok());
        assert_eq!(p.layers().len(), 28);
        assert_eq!(p.state().layers, 28);
        assert!(matches!(
            p.layers()[0].operators[0],
            Operator::RmsNorm { .. }
        ));
        assert!(matches!(
            p.layers()[0].operators.last(),
            Some(Operator::ResidualAdd)
        ));
        assert_eq!(
            p.layers()[0]
                .operators
                .iter()
                .filter(|o| matches!(o, Operator::HeadRmsNorm { .. }))
                .count(),
            usize::from(family == Architecture::Qwen3)
        );
        for layer in p.layers() {
            for op in &layer.operators {
                if let Operator::Linear { weight, .. } | Operator::RmsNorm { weight, .. } = op {
                    assert!(p.tensor(weight).is_some(), "{weight}");
                }
            }
        }
    }
}
fn manifest(values: &[(&str, shimmytok::gguf::GGUFValue)]) -> shimmytok::manifest::ModelManifest {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend((values.len() as u64).to_le_bytes());
    for (key, value) in values {
        b.extend((key.len() as u64).to_le_bytes());
        b.extend(key.as_bytes());
        match value {
            shimmytok::gguf::GGUFValue::String(s) => {
                b.extend(8u32.to_le_bytes());
                b.extend((s.len() as u64).to_le_bytes());
                b.extend(s.as_bytes());
            }
            shimmytok::gguf::GGUFValue::U32(x) => {
                b.extend(4u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            _ => unreachable!(),
        }
    }
    shimmytok::manifest::read_manifest(std::io::Cursor::new(&b), b.len() as u64, Default::default())
        .unwrap()
}
#[test]
fn unsupported_families_cannot_become_dense_plans() {
    use shimmytok::gguf::GGUFValue::String as S;
    for family in [
        "qwen3moe",
        "deepseek2",
        "qwen3next",
        "qwen35",
        "mamba",
        "gemma3",
        "minicpmv",
        "unknown",
    ] {
        let m = manifest(&[("general.architecture", S(family.into()))]);
        assert_eq!(
            ValidatedModelDescriptor::from_manifest(&m)
                .unwrap_err()
                .code,
            "unsupported_architecture"
        );
    }
}
#[test]
fn missing_required_metadata_has_typed_error() {
    let m = manifest(&[(
        "general.architecture",
        shimmytok::gguf::GGUFValue::String("qwen3".into()),
    )]);
    let e = ValidatedModelDescriptor::from_manifest(&m).unwrap_err();
    assert_eq!(e.code, "missing_metadata");
    assert_eq!(e.subject, "qwen3.embedding_length");
}
#[test]
fn dense_named_moe_cannot_alias_dense_execution() {
    use shimmytok::gguf::GGUFValue::{String as S, U32};
    let m = manifest(&[
        ("general.architecture", S("llama".into())),
        ("llama.expert_count", U32(8)),
    ]);
    assert_eq!(
        ValidatedModelDescriptor::from_manifest(&m)
            .unwrap_err()
            .code,
        "unsupported_execution_requirement"
    );
}
