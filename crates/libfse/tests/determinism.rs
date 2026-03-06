// crates/libfse/tests/determinism.rs

#[cfg(test)]
mod tests {
    use libfse::{FseMap, FseOpcode, FseScanner, Rule};

    // The Determinism Gate
    // Ensures that repeated execution on the same scanner instance yields identical results.
    // This catches "sticky bit" bugs where `reset_rule_state` is incomplete.
    #[test]
    fn test_execution_stability() {
        let rules = vec![
            Rule::new("apple", FseOpcode::Record(1)),
            Rule::new("banana", FseOpcode::Record(2)),
            Rule::new("cherry", FseOpcode::Ignore), 
        ];

        let map = FseMap::compile(rules).unwrap();
        let mut scanner = FseScanner::new(&map).unwrap();

        let input = b"I like apple and banana but not cherry.";

        // Run 1: Cold Start
        let r1 = scanner.scan(input).unwrap();
        
        // Assert Baseline properties
        assert_eq!(r1.rules_recorded, 2, "Should record apple(1) and banana(2)");

        // Run 2..1000: Hot Loop Resets
        for i in 2..=1000 {
            scanner.reset_rule_state();
            scanner.reset_automaton_state().unwrap();

            let ri = scanner.scan(input).unwrap();

            // Strict Equality Check
            assert_eq!(ri.rules_recorded, r1.rules_recorded, 
                "Divergence at iteration {}: Expected {} recorded rules, got {}", 
                i, r1.rules_recorded, ri.rules_recorded
            );
            
            assert_eq!(ri.match_states_seen, r1.match_states_seen,
                "Divergence at iteration {}: Match states count differed", i
            );
        }
    }
}
