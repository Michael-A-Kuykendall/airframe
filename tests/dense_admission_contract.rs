use airframe::execution::admission::{inspect_model, PreparedDenseModel};
#[test]
fn malformed_models_reject_without_gpu_initialization() {
    let mut bytes = Vec::from(*b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), bytes).unwrap();
    let inspection = inspect_model(file.path()).unwrap();
    assert!(!inspection.structural_support);
    assert_eq!(inspection.reason_code.as_deref(), Some("missing_metadata"));
    let error = match PreparedDenseModel::load(file.path(), Some(32)) {
        Ok(_) => panic!("invalid model admitted"),
        Err(e) => e,
    };
    assert_eq!(error.code(), "missing_metadata");
}
#[test]
#[ignore = "requires explicitly supplied private fixture; absence is NOT_RUN"]
fn real_qwen_manifest_plan_and_device_contract() {
    let path = std::path::PathBuf::from(
        std::env::var("AIRFRAME_QWEN_FIXTURE").expect("set exact fixture"),
    );
    let prepared = PreparedDenseModel::load(&path, Some(512)).unwrap();
    let model = prepared.validated();
    assert_eq!(model.state_spec().capacity, 512);
    assert_eq!(
        model.identity().model_sha256,
        "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"
    );
    assert!(model.check_forward(model.spec(), 8, 0, 8).is_ok());
    assert!(model.check_forward(model.spec(), 8, 0, 7).is_err());
    assert!(model.check_forward(model.spec(), 1, 512, 513).is_err());
    let mut wrong = model.spec().clone();
    wrong.rope_base *= 2.0;
    assert!(model.check_spec(&wrong).is_err());
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 14;
    limits.max_compute_invocations_per_workgroup = 256;
    model.check_device(&limits).unwrap();
    limits.max_storage_buffers_per_shader_stage = 8;
    assert!(model.check_device(&limits).is_err());
    assert!(PreparedDenseModel::load(&path, Some(0)).is_err());
    assert!(PreparedDenseModel::load(&path, Some(usize::MAX)).is_err());
    let report = inspect_model(&path).unwrap();
    assert!(report.structural_support);
    assert_eq!(report.numerical_certification, "NOT_RUN");
    assert_eq!(report.platform_certification, "NOT_RUN");
}
