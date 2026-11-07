#[test]
#[ignore = "requires exact private GPU probe binary and Qwen fixture; otherwise NOT_RUN"]
fn request_failures_cancel_and_reset_cannot_reuse_invalid_state() {
    let binary = std::env::var("AIRFRAME_STATE_PROBE_BIN")
        .expect("set the private, source-pinned main-thread probe binary");
    let output = std::process::Command::new(binary)
        .output()
        .expect("start GPU probe");
    assert!(
        output.status.success(),
        "GPU probe failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("STATE_PROBE_PASS"));
}
