use airframe_observe::{
    facts::HaltReason,
    isf::{ISFState, InferenceSaturationFabric},
};
use std::sync::{Arc, Mutex};

fn run(ids: Vec<u32>, max: u32, stops: Vec<u32>) -> (String, usize, HaltReason, usize) {
    let state = Arc::new(Mutex::new(ISFState::new(1, max, 0, stops, None)));
    {
        let mut s = state.lock().unwrap();
        s.all_token_ids = vec![7];
        s.prompt_token_ids = vec![7];
    }
    let sequence = Arc::new(Mutex::new((ids.into_iter(), 0usize)));
    let sampled = sequence.clone();
    let mut fabric = InferenceSaturationFabric::new(
        state.clone(),
        Arc::new(|_, dim| vec![1.0; dim as usize]),
        Arc::new(|_, _| (vec![1.0; 2], vec![1.0; 10])),
        Arc::new(|_, _| (vec![1.0; 2], vec![1.0; 10])),
        Arc::new(move |_, _| {
            let mut seq = sampled.lock().unwrap();
            seq.1 += 1;
            seq.0
                .next()
                .expect("must not sample beyond the requested budget")
        }),
        Arc::new(|id| id.to_string()),
        Arc::new(|| {}),
        2,
    );
    let result = fabric.generate(&[7]);
    let samples = sequence.lock().unwrap().1;
    let halt = state
        .lock()
        .unwrap()
        .halt
        .clone()
        .expect("terminal reason retained");
    (result.text, result.tokens_generated, halt, samples)
}

#[test]
fn output_budget_is_exact_without_an_extra_sample() {
    for max in [1, 2, 4] {
        let (text, count, reason, samples) = run(vec![1, 2, 3, 4], max, vec![]);
        assert_eq!(count, max as usize);
        assert_eq!(samples, max as usize);
        assert_eq!(text.chars().count(), max as usize);
        assert!(matches!(reason, HaltReason::MaxTokensReached));
    }
}

#[test]
fn terminal_eos_is_counted_but_not_emitted() {
    for ids in [vec![0], vec![1, 0]] {
        let expected_count = ids.len();
        let (text, count, reason, samples) = run(ids, 8, vec![]);
        assert_eq!(count, expected_count);
        assert_eq!(samples, expected_count);
        assert!(!text.contains('0'));
        assert!(matches!(reason, HaltReason::EosToken));
    }
}

#[test]
fn extra_stop_has_its_own_terminal_reason() {
    for ids in [vec![9], vec![1, 9]] {
        let expected_count = ids.len();
        let (text, count, reason, _) = run(ids, 8, vec![9]);
        assert_eq!(count, expected_count);
        assert!(!text.contains('9'));
        assert!(matches!(reason, HaltReason::ExtraStopToken));
    }
}
