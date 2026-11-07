use airframe::execution::admission::{validate_kernel_configuration, PreparedDenseModel};

const FLAGS: &[&str] = &[
    "AIRFRAME_EXPERIMENTAL_STAGED_FFN",
    "AIRFRAME_PINGPONG_ACTIVATION",
    "AIRFRAME_EXPERIMENTAL_COOPERATIVE_FFN_DOWN",
    "AIRFRAME_EXPERIMENTAL_COOPERATIVE_QKV",
    "AIRFRAME_EXPERIMENTAL_COOPERATIVE_ATTN_PROJ",
    "AIRFRAME_EXPERIMENTAL_KV_ZERO_UPLOAD",
    "AIRFRAME_EXPERIMENTAL_VECTORIZED_FFN",
    "SHIMMY_DISABLE_OUTPUT_NORM",
    "SHIMMY_ROUTE_V2_LAYER_PARAMS",
    "SHIMMY_KV_QUANT",
    "AIRFRAME_KV_QUANT",
    "SHIMMY_ROPE_SCALE",
];
fn check_flag(flag: &str) {
    for value in ["1", "true", "TRUE", "0"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        for name in FLAGS {
            command.env_remove(name);
        }
        let output = command
            .args(["--exact", "admission_flag_child", "--nocapture"])
            .env("AIRFRAME_ADMISSION_TEST_FLAG", flag)
            .env("AIRFRAME_ADMISSION_TEST_VALUE", value)
            .env(flag, value)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{flag}={value}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
#[test]
fn vectorized_ffn_rejects_before_model_io() {
    check_flag("AIRFRAME_EXPERIMENTAL_VECTORIZED_FFN");
}
#[test]
fn output_norm_bypass_rejects_before_model_io() {
    check_flag("SHIMMY_DISABLE_OUTPUT_NORM");
}
#[test]
fn route_v2_params_reject_before_model_io() {
    check_flag("SHIMMY_ROUTE_V2_LAYER_PARAMS");
}
#[test]
fn admission_flag_child() {
    let Ok(flag) = std::env::var("AIRFRAME_ADMISSION_TEST_FLAG") else {
        return;
    };
    let value = std::env::var("AIRFRAME_ADMISSION_TEST_VALUE").unwrap();
    if value == "0" {
        validate_kernel_configuration().unwrap();
        return;
    }
    let error = validate_kernel_configuration().unwrap_err();
    assert_eq!(error.code(), "unsupported_kernel_configuration");
    assert!(error.to_string().contains(&flag));
    let error = match PreparedDenseModel::load(
        std::path::Path::new("/nonexistent-audit-fixture.gguf"),
        Some(8),
    ) {
        Ok(_) => panic!("unvalidated configuration admitted"),
        Err(e) => e,
    };
    assert_eq!(error.code(), "unsupported_kernel_configuration");
    assert!(error.to_string().contains(&flag));
}
