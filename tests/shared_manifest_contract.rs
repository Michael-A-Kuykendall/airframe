use airframe::backend::bindless::metadata::BindlessMetadata;
use std::io::Cursor;
fn file(alignment: u32) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    let key = "general.alignment";
    b.extend((key.len() as u64).to_le_bytes());
    b.extend(key.as_bytes());
    b.extend(4u32.to_le_bytes());
    b.extend(alignment.to_le_bytes());
    let name = "output_norm.weight";
    b.extend((name.len() as u64).to_le_bytes());
    b.extend(name.as_bytes());
    b.extend(1u32.to_le_bytes());
    b.extend(8u64.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    let a = if alignment.is_power_of_two() {
        alignment as usize
    } else {
        32
    };
    b.resize(((b.len() + a - 1) & !(a - 1)) + 32, 0);
    b
}
#[test]
fn adapter_honors_declared_alignment() {
    let bytes = file(64);
    let len = bytes.len() as u64;
    let manifest =
        shimmytok::manifest::read_manifest(Cursor::new(&bytes), len, Default::default()).unwrap();
    let a = BindlessMetadata::try_new(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(a.data_start_offset, manifest.data_start());
    assert_eq!(
        a.get_tensor_offset("output_norm.weight"),
        Some(
            manifest
                .tensor("output_norm.weight")
                .unwrap()
                .absolute_offset()
        )
    );
    assert_eq!(a.tensor_dims["output_norm.weight"], vec![8]);
}
#[test]
fn invalid_manifest_is_error_before_compilation() {
    assert!(BindlessMetadata::try_new(&mut Cursor::new(file(48))).is_err());
    let bytes = file(64);
    for len in 0..80 {
        assert!(BindlessMetadata::try_new(&mut Cursor::new(&bytes[..len])).is_err());
    }
}
#[test]
#[ignore = "requires exact local GGUF fixtures"]
fn real_model_offsets_match_authoritative_manifest() {
    let root = std::env::var("SHIMMY_CERT_MODEL_DIR").expect("fixtures");
    for name in [
        "Qwen3-0.6B-Q4_K_M.gguf",
        "llama3.2-3b-instruct-dde5aa3f.gguf",
    ] {
        let path = std::path::Path::new(&root).join(name);
        let m = shimmytok::manifest::load_manifest(&path).unwrap();
        let a = BindlessMetadata::try_new(&mut std::fs::File::open(path).unwrap()).unwrap();
        assert_eq!(a.tensor_count, m.tensors().len() as u64);
        for t in m.tensors() {
            assert_eq!(a.get_tensor_offset(t.name()), Some(t.absolute_offset()));
            assert_eq!(a.get_tensor_type(t.name()), Some(t.ggml_type()));
            assert_eq!(a.tensor_dims[t.name()], t.dimensions());
        }
        assert_eq!(a.compiled_layers.len(), 28);
    }
}
