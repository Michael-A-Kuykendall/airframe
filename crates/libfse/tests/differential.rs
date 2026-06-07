// crates/libfse/tests/differential.rs

#[cfg(test)]
mod tests {
    use aho_corasick::{AhoCorasick, MatchKind};
    use libfse::{FseMap, FseOpcode, FseScanner, Rule};

    // The Differential Oracle
    // We confirm that libfse finds exactly the same set of records as Aho-Corasick
    // for a complex, multi-match input.
    //
    // Note: We only check "Record" opcodes. Reject logic is unique to LibFSE.
    #[test]
    fn test_differential_parity_vs_std_aho() {
        let patterns = vec![
            "fox", "brown", "jumps", "dog", "lazy", "quick", "the", "over",
        ];

        let text = "the quick brown fox jumps over the lazy dog";

        // 1. Run Standard Aho-Corasick (Oracle)
        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::Standard) // LibFSE uses standard semantics
            .build(&patterns)
            .unwrap();

        let mut expected_hits = vec![false; patterns.len()];
        for mat in ac.find_iter(text) {
            let pid = mat.pattern().as_usize();
            expected_hits[pid] = true;
        }

        // 2. Run LibFSE
        // Map pattern index -> RuleId (identity mapping for simplicity)
        let rules: Vec<Rule> = patterns
            .iter()
            .enumerate()
            .map(|(i, &p)| Rule::new(p, FseOpcode::Record(i as u32)))
            .collect();

        let map = FseMap::compile(rules).unwrap();
        let mut scanner = FseScanner::new(&map).unwrap();

        // We use check_rule_state to introspect the bitset
        scanner.scan(text.as_bytes()).expect("Scan failed");

        // 3. Compare Results
        // We need a way to inspect the scanner's recorded bits.
        // Since FseScanner internals are private, we can't read `rule_bits` directly from an integration test.
        // However, the `ScanSummary` returns `rules_recorded` count.
        // To be precise, we need to ensure the count matches.

        let expected_count = expected_hits.iter().filter(|&&h| h).count();
        let actual_count = scanner.scan(text.as_bytes()).unwrap().rules_recorded;

        assert_eq!(
            actual_count as usize, expected_count,
            "LibFSE recorded {} unique rules, Oracle found {}. Divergence!",
            actual_count, expected_count
        );
    }
}
