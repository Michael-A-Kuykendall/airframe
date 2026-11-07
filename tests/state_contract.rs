use airframe::runtime::state::*;
fn controller() -> std::sync::Arc<StateController> {
    StateController::new(
        StateIdentity {
            model_sha256: "model".into(),
            plan_sha256: "plan".into(),
        },
        StateSpec {
            kind: StateKind::DenseKv,
            layers: 2,
            kv_heads: 2,
            head_dim: 8,
            capacity: 8,
            precision: KvPrecision::F32,
        },
    )
    .unwrap()
}
#[test]
fn state_contract_commits_only_complete_gpu_writes() {
    let c = controller();
    let mut s = c.begin().unwrap();
    s.start_prefill(3).unwrap();
    let w = s.prepare_prefill_chunk(0, 2).unwrap();
    assert!(s.require_read(0, 1).is_err());
    s.commit(w).unwrap();
    assert!(s.require_read(0, 2).is_ok());
    assert!(s.prepare_decode(2).is_err());
    let w = s.prepare_prefill_chunk(2, 1).unwrap();
    s.commit(w).unwrap();
    let w = s.prepare_decode(3).unwrap();
    assert_eq!(s.snapshot().processed_positions, 3);
    s.commit(w).unwrap();
    assert_eq!(s.snapshot().processed_positions, 4);
    s.finish().unwrap();
    assert!(s.require_read(0, 1).is_err());
}
#[test]
fn state_contract_cancellation_and_repeated_requests_are_isolated() {
    let c = controller();
    let first = {
        let mut s = c.begin().unwrap();
        let epoch = s.snapshot().epoch;
        s.start_prefill(2).unwrap();
        let _w = s.prepare_prefill_chunk(0, 2).unwrap();
        assert!(c.begin().is_err());
        s.abort();
        epoch
    };
    let s = c.begin().unwrap();
    assert_eq!(s.snapshot().processed_positions, 0);
    assert!(s.snapshot().epoch > first);
    assert!(s
        .check_identity(&StateIdentity {
            model_sha256: "other".into(),
            plan_sha256: "plan".into()
        })
        .is_err());
}
#[test]
fn state_contract_rejects_overflow_invalid_ranges_and_unsupported_operations() {
    let c = controller();
    let mut s = c.begin().unwrap();
    assert!(s.start_prefill(9).is_err());
    s.start_prefill(2).unwrap();
    assert!(s.prepare_prefill_chunk(usize::MAX, 2).is_err());
    assert!(s.rollback(0).is_err());
    assert!(s.finish().is_err());
    let wrong = StateSpec {
        kind: StateKind::Mamba,
        layers: 1,
        kv_heads: 1,
        head_dim: 8,
        capacity: 8,
        precision: KvPrecision::F32,
    };
    assert!(wrong.budget(1024, 1024).is_err());
    let huge = StateSpec {
        kind: StateKind::DenseKv,
        layers: usize::MAX,
        kv_heads: usize::MAX,
        head_dim: 128,
        capacity: usize::MAX,
        precision: KvPrecision::F32,
    };
    assert!(huge.budget(u64::MAX, u64::MAX).is_err());
}
#[test]
fn state_contract_budget_includes_existing_int4_shadow_and_scales() {
    let mut spec = StateSpec {
        kind: StateKind::DenseKv,
        layers: 2,
        kv_heads: 2,
        head_dim: 8,
        capacity: 8,
        precision: KvPrecision::F32,
    };
    assert_eq!(spec.budget(1024, 4096).unwrap().total_bytes, 2048);
    spec.precision = KvPrecision::F32WithInt4Shadow;
    assert_eq!(spec.budget(1024, 4096).unwrap().total_bytes, 2560);
    assert!(spec.budget(511, 4096).is_err());
}
