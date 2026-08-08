//! Capture protocol test — verifies the capture protocol types work correctly.

use airframe_conformance::capture_protocol::*;

#[test]
fn point_identity_serialization() {
    let id = PointIdentity {
        layer: 1,
        position: 0,
        stage: "attn_q".to_string(),
        sub_stage: Some("pre_rope".to_string()),
    };

    let json = serde_json::to_string(&id).unwrap();
    let parsed: PointIdentity = serde_json::from_str(&json).unwrap();
    assert_eq!(id, parsed);
}

#[test]
fn point_identity_minimal() {
    let id = PointIdentity {
        layer: 0,
        position: 0,
        stage: "embedding".to_string(),
        sub_stage: None,
    };

    let json = serde_json::to_string(&id).unwrap();
    let parsed: PointIdentity = serde_json::from_str(&json).unwrap();
    assert_eq!(id, parsed);
    assert!(!json.contains("sub_stage"));
}

#[test]
fn availability_serialization() {
    let avail = Availability::Real;
    let json = serde_json::to_string(&avail).unwrap();
    assert_eq!(json, "\"real\"");

    let avail = Availability::Proxy;
    let json = serde_json::to_string(&avail).unwrap();
    assert_eq!(json, "\"proxy\"");

    let avail = Availability::Unavailable;
    let json = serde_json::to_string(&avail).unwrap();
    assert_eq!(json, "\"unavailable\"");

    let avail = Availability::Error("test error".to_string());
    let json = serde_json::to_string(&avail).unwrap();
    assert!(json.contains("test error"));
}

#[test]
fn coordinate_plan_serialization() {
    let coord = CoordinatePlan {
        binding: 0,
        offset: 1024,
        count: 2560,
        stride: 4,
        format: "f32".to_string(),
        shape: vec![2560],
    };

    let json = serde_json::to_string(&coord).unwrap();
    let parsed: CoordinatePlan = serde_json::from_str(&json).unwrap();
    assert_eq!(coord, parsed);
}

#[test]
fn provenance_serialization() {
    let prov = Provenance {
        engine: "airframe_product".to_string(),
        engine_version: "0.3.0".to_string(),
        captured_at: "2026-01-01T00:00:00Z".to_string(),
        git_commit: Some("abc123".to_string()),
        build_profile: "release".to_string(),
        config_hash: "def456".to_string(),
    };

    let json = serde_json::to_string(&prov).unwrap();
    let parsed: Provenance = serde_json::from_str(&json).unwrap();
    assert_eq!(prov, parsed);
}

#[test]
fn tensor_stats_serialization() {
    let stats = TensorStats {
        rms: 1.23,
        nan_count: 0,
        first8: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        sampled: SampledSource::Real,
    };

    let json = serde_json::to_string(&stats).unwrap();
    let parsed: TensorStats = serde_json::from_str(&json).unwrap();
    assert_eq!(stats, parsed);
}

#[test]
fn capture_point_with_data() {
    let point = CapturePoint {
        identity: PointIdentity {
            layer: 1,
            position: 0,
            stage: "residual_pre".to_string(),
            sub_stage: None,
        },
        availability: Availability::Real,
        coordinate: CoordinatePlan {
            binding: 1,
            offset: 0,
            count: 2560,
            stride: 4,
            format: "f32".to_string(),
            shape: vec![2560],
        },
        provenance: Provenance {
            engine: "airframe_product".to_string(),
            engine_version: "0.3.0".to_string(),
            captured_at: "2026-01-01T00:00:00Z".to_string(),
            git_commit: Some("abc123".to_string()),
            build_profile: "release".to_string(),
            config_hash: "def456".to_string(),
        },
        stats: Some(TensorStats {
            rms: 1.0,
            nan_count: 0,
            first8: vec![0.1; 8],
            sampled: SampledSource::Real,
        }),
        data: Some("base64data".to_string()),
    };

    let json = serde_json::to_string(&point).unwrap();
    let parsed: CapturePoint = serde_json::from_str(&json).unwrap();
    assert_eq!(point, parsed);
}

#[test]
fn capture_point_unavailable() {
    let point = CapturePoint {
        identity: PointIdentity {
            layer: 1,
            position: 0,
            stage: "attn_q".to_string(),
            sub_stage: None,
        },
        availability: Availability::Unavailable,
        coordinate: CoordinatePlan {
            binding: 1,
            offset: 0,
            count: 2560,
            stride: 4,
            format: "f32".to_string(),
            shape: vec![2560],
        },
        provenance: Provenance {
            engine: "airframe_product".to_string(),
            engine_version: "0.3.0".to_string(),
            captured_at: "2026-01-01T00:00:00Z".to_string(),
            git_commit: None,
            build_profile: "release".to_string(),
            config_hash: "def456".to_string(),
        },
        stats: None,
        data: None,
    };

    let json = serde_json::to_string(&point).unwrap();
    let parsed: CapturePoint = serde_json::from_str(&json).unwrap();
    assert_eq!(point, parsed);
    assert!(!json.contains("stats"));
    assert!(!json.contains("data"));
}

#[test]
fn model_config_serialization() {
    let config = ModelConfig {
        arch: "llama".to_string(),
        n_layer: 32,
        n_embd: 4096,
        n_head: 32,
        n_kv_head: 8,
        head_dim: 128,
        rope_base: Some(1000000.0),
        rope_dim: Some(128),
        rms_eps: Some(1e-6),
        qk_norm: Some(false),
        n_ctx_capped: Some(8192),
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config, parsed);
}

#[test]
fn model_config_minimal() {
    let config = ModelConfig {
        arch: "qwen3".to_string(),
        n_layer: 36,
        n_embd: 4096,
        n_head: 32,
        n_kv_head: 8,
        head_dim: 128,
        rope_base: None,
        rope_dim: None,
        rms_eps: None,
        qk_norm: None,
        n_ctx_capped: None,
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config, parsed);
    // Optional fields should not appear when None
    assert!(!json.contains("rope_base"));
    assert!(!json.contains("rope_dim"));
    assert!(!json.contains("rms_eps"));
    assert!(!json.contains("qk_norm"));
    assert!(!json.contains("n_ctx_capped"));
}

#[test]
fn prompt_capture_serialization() {
    let capture = PromptCapture {
        prompt: "The capital of France is".to_string(),
        token_ids: vec![123, 456, 789, 101],
        token_pieces: Some(vec![
            "The".to_string(),
            " capital".to_string(),
            " of".to_string(),
            " France".to_string(),
        ]),
        points: vec![],
        model_config: ModelConfig {
            arch: "llama".to_string(),
            n_layer: 32,
            n_embd: 4096,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            rope_base: Some(1000000.0),
            rope_dim: Some(128),
            rms_eps: Some(1e-6),
            qk_norm: Some(false),
            n_ctx_capped: Some(8192),
        },
    };

    let json = serde_json::to_string(&capture).unwrap();
    let parsed: PromptCapture = serde_json::from_str(&json).unwrap();
    assert_eq!(capture, parsed);
}

#[test]
fn capture_manifest_serialization() {
    let manifest = CaptureManifest::new("run-123".to_string(), "0.1.0".to_string(), vec![]);

    assert_eq!(manifest.schema, "airframe.conformance.capture.v1");
    assert_eq!(manifest.run_id, "run-123");
    assert_eq!(manifest.conformance_version, "0.1.0");

    let json = serde_json::to_string(&manifest).unwrap();
    let parsed: CaptureManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(manifest, parsed);
}

#[test]
fn capture_manifest_validates() {
    let manifest = CaptureManifest::new(
        "run-123".to_string(),
        "0.1.0".to_string(),
        vec![PromptCapture {
            prompt: "test".to_string(),
            token_ids: vec![1],
            token_pieces: None,
            points: vec![],
            model_config: ModelConfig {
                arch: "test".to_string(),
                n_layer: 1,
                n_embd: 1,
                n_head: 1,
                n_kv_head: 1,
                head_dim: 1,
                rope_base: None,
                rope_dim: None,
                rms_eps: None,
                qk_norm: None,
                n_ctx_capped: None,
            },
        }],
    );

    // Should not panic
    manifest
        .validate()
        .expect("Capture manifest should validate");
}

#[test]
fn sampled_source_serialization() {
    let sources = [
        SampledSource::Real,
        SampledSource::Proxy,
        SampledSource::Unavailable,
        SampledSource::TempLast,
        SampledSource::ActivationLast,
        SampledSource::WrongBuffer,
        SampledSource::ProxyResidual,
    ];

    for source in sources {
        let json = serde_json::to_string(&source).unwrap();
        let parsed: SampledSource = serde_json::from_str(&json).unwrap();
        assert_eq!(source, parsed);
    }
}
