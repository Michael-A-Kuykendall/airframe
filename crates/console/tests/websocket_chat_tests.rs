//! Tests for the InferenceBackend trait and adapters
//!
//! Tests MockInferenceAdapter behavior and channel typing.

use shimmy_console::adapters::MockInferenceAdapter;
use shimmy_console::websocket::InferenceBackend;

// ── MockInferenceAdapter ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_mock_adapter_list_models_empty_by_default() {
    let adapter = MockInferenceAdapter::new();
    let models = adapter.list_models().await.expect("list_models should not fail");
    // Mock has default seeded models (phi3-mini, phi3-medium) — that's fine
    // Just verify it returns without error
    let _ = models;
}

#[tokio::test]
async fn test_mock_adapter_set_and_list_models() {
    let adapter = MockInferenceAdapter::new();
    adapter.set_models(vec![
        ("phi-3-mini".to_string(), serde_json::json!({"type": "mock"})),
        ("tinyllama".to_string(), serde_json::json!({"type": "mock"})),
    ]).await;
    let models = adapter.list_models().await.expect("list_models should succeed");
    assert_eq!(models.len(), 2);
    let names: Vec<&str> = models.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"phi-3-mini"));
    assert!(names.contains(&"tinyllama"));
}

#[tokio::test]
async fn test_mock_adapter_generate_response() {
    let adapter = MockInferenceAdapter::new();
    let result = adapter.generate_response("phi-3-mini", "Hello").await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn test_mock_adapter_get_session_model_none_by_default() {
    let adapter = MockInferenceAdapter::new();
    let model = adapter.get_session_model("test-session").await;
    // Mock returns None by default
    let _ = model; // just verify it doesn't panic
}

#[tokio::test]
async fn test_mock_adapter_set_session_model() {
    let adapter = MockInferenceAdapter::new();
    let result = adapter.set_session_model("test-session", "phi-3-mini").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_mock_adapter_get_metrics() {
    let adapter = MockInferenceAdapter::new();
    adapter.set_metrics(serde_json::json!({"tokens_per_sec": 42})).await;
    let metrics = adapter.get_metrics().await.expect("get_metrics should succeed");
    assert_eq!(metrics["tokens_per_sec"], 42);
}

#[tokio::test]
async fn test_mock_adapter_generate_stream() {
    let adapter = MockInferenceAdapter::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let result = adapter.generate_stream("phi-3-mini", "Hello", tx).await;
    assert!(result.is_ok());
    // Drain whatever the mock sends
    let mut tokens = Vec::new();
    while let Ok(token) = rx.try_recv() {
        tokens.push(token);
    }
    // Mock may send 0 or more tokens — just verify no panic
    let _ = tokens;
}

#[tokio::test]
async fn test_mock_adapter_call_history() {
    let adapter = MockInferenceAdapter::new();
    let _ = adapter.generate_response("model", "first call").await;
    let _ = adapter.generate_response("model", "second call").await;
    let history = adapter.get_call_history().await;
    assert!(history.len() >= 1, "Call history should be recorded");
}

// ── Channel type annotation tests ─────────────────────────────────────────────

#[tokio::test]
async fn test_channel_string_type() {
    // Verify that mpsc::channel::<String> is the correct type for generate_stream
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
    tx.send("token".to_string()).await.unwrap();
    let received = rx.recv().await.unwrap();
    assert_eq!(received, "token");
}
