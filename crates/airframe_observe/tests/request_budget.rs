use airframe_observe::facts::{HaltReason, InferenceFact};
use airframe_observe::isf::{
    d0_run_budget, request_run_budget, ISFState, InferenceSaturationFabric, IsfRunError,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

fn fixture(count: usize, unique: bool) -> (InferenceSaturationFabric, Vec<u32>, Arc<AtomicUsize>) {
    let ids: Vec<u32> = (0..count)
        .map(|i| if unique { (i + 1) as u32 } else { 7 })
        .collect();
    let state = Arc::new(Mutex::new(ISFState::new(count as u32, 16, 0, vec![], None)));
    {
        let mut s = state.lock().unwrap();
        s.prompt_token_ids = ids.clone();
        s.all_token_ids = ids.clone();
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let sink = calls.clone();
    let fabric = InferenceSaturationFabric::new(
        state,
        Arc::new(|_, dim| vec![1.; dim as usize]),
        Arc::new(move |_, _| {
            sink.fetch_add(1, Ordering::SeqCst);
            (vec![1.; 2], vec![1.; 10])
        }),
        Arc::new(|_, _| (vec![1.; 2], vec![1.; 10])),
        Arc::new(|_, _| 1),
        Arc::new(|_| "x".into()),
        Arc::new(|| {}),
        2,
    );
    (fabric, ids, calls)
}

#[test]
fn old_default_stops_before_long_prompt_can_prefill() {
    let (mut fabric, ids, calls) = fixture(2032, false);
    for (position, &token_id) in ids.iter().enumerate() {
        fabric.fabric.assert(InferenceFact::PromptToken {
            position: position as u32,
            token_id,
        });
    }
    let run = fabric.fabric.run_to_fixpoint(d0_run_budget());
    assert!(!run.saturated);
    assert_eq!(run.total_rounds, 1000);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(fabric.state.lock().unwrap().halt.is_none());
}

#[test]
fn bounded_request_budget_completes_repeated_and_unique_long_prompts() {
    for (count, unique) in [(2032, false), (2032, true), (8176, false), (8176, true)] {
        let (mut fabric, ids, calls) = fixture(count, unique);
        let result = fabric.try_generate(&ids).unwrap();
        assert_eq!(result.tokens_generated, 16, "count={count} unique={unique}");
        assert_eq!(result.halt_reason, HaltReason::MaxTokensReached);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn request_budget_overflow_is_rejected_before_fact_insertion() {
    assert_eq!(
        request_run_budget(usize::MAX, 1, 0).unwrap_err(),
        IsfRunError::BudgetOverflow
    );
    assert_eq!(
        request_run_budget(1, usize::MAX, 0).unwrap_err(),
        IsfRunError::BudgetOverflow
    );
    assert_eq!(
        request_run_budget(1, 1, usize::MAX).unwrap_err(),
        IsfRunError::BudgetOverflow
    );
}

#[test]
fn runaway_rules_return_typed_budget_exhaustion_instead_of_partial_success() {
    use dzero::{AlphaKey, ClosureProgram, SaturationFabric};
    let (mut fabric, ids, _) = fixture(1, false);
    let mut program = ClosureProgram::new();
    program.register(AlphaKey(1), |fact, _| match fact {
        InferenceFact::PromptToken { position, token_id } => vec![InferenceFact::PromptToken {
            position: position + 1,
            token_id: *token_id,
        }],
        _ => vec![],
    });
    fabric.fabric = SaturationFabric::new(program, |_| Some(AlphaKey(1)), |_, _| vec![]);
    assert!(matches!(
        fabric.try_generate(&ids),
        Err(IsfRunError::BudgetExhausted)
    ));
}
