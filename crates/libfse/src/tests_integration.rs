#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    // use super::*; // Unused
    use crate::{FseMap, FseOpcode, FseScanner, Rule, Violation};

    #[test]
    fn test_basic_match_record() {
        let rules = vec![
            Rule::new("apple", FseOpcode::Record(1)),
            Rule::new("banana", FseOpcode::Record(2)),
        ];
        let map = FseMap::compile(rules).unwrap();
        let mut scanner = FseScanner::new(&map).unwrap();

        let input = b"I like apple and banana splits.";
        let summary = scanner.scan(input).unwrap();

        assert_eq!(summary.pattern_hits, 2);
        assert_eq!(summary.rules_recorded, 2);
    }

    #[test]
    fn test_fail_closed_reject() {
        let rules = vec![
            Rule::new("safe", FseOpcode::Record(1)),
            Rule::new("DANGER", FseOpcode::Reject(99)),
        ];
        let map = FseMap::compile(rules).unwrap();
        let mut scanner = FseScanner::new(&map).unwrap();

        let input = b"This is safe but DANGER resides here.";
        let result = scanner.scan(input);

        match result {
            Err(Violation::PolicyReject { rule_id, span, .. }) => {
                assert_eq!(rule_id, 99);
                // "DANGER" is 6 bytes.
                // Input: "This is safe but DANGER..."
                // Index:  012345678901234567890123456
                // "T" is 0. "D" is at 17. "R" is at 22. End is 23.
                // Let's just check the byte slice matches.
                assert_eq!(&input[span], b"DANGER");
            }
            _ => panic!("Should have rejected!"),
        }
    }

    #[test]
    fn test_overlapping_matches() {
        // "he" -> Record(1)
        // "she" -> Record(2)
        // "hers" -> Record(3)
        let rules = vec![
            Rule::new("he", FseOpcode::Record(1)),
            Rule::new("she", FseOpcode::Record(2)),
            Rule::new("hers", FseOpcode::Record(3)),
        ];
        let map = FseMap::compile(rules).unwrap();
        let mut scanner = FseScanner::new(&map).unwrap();

        let input = b"ushers";
        let summary = scanner.scan(input).unwrap();

        // "she" matches at "..she.."
        // "he" matches inside "she" and in "he" of "hers"?
        // Aho-Corasick behavior depends on how we iterate.
        // We iterate ALL matches at a state.

        // "she": Matches "she". Contains "he".
        // "hers": Matches "hers". Contains "he".

        // We expect multiple hits.
        assert!(summary.pattern_hits >= 3);
    }
}
