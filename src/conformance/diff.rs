use crate::conformance::fixtures::ConformanceFixture;
use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Result of conformance comparison
#[derive(Debug, Clone)]
pub struct ConformanceDiff {
    pub passed: bool,
    pub token_match: bool,
    pub topk_overlap: TopKOverlap,
    pub details: Vec<String>,
}

/// Top-K overlap analysis
#[derive(Debug, Clone)]
pub struct TopKOverlap {
    pub prefill_overlap: usize,
    pub prefill_total: usize,
    pub decode_overlap: usize,
    pub decode_total: usize,
    pub prefill_ratio: f32,
    pub decode_ratio: f32,
}

/// Actual inference results to compare against fixture
#[derive(Debug, Clone)]
pub struct InferenceResults {
    pub prefill_logits: Tensor,
    pub decode_token: usize,
    pub decode_logits: Tensor,
}

/// Compare inference results against conformance fixture
pub fn diff_conformance(
    fixture: &ConformanceFixture,
    results: &InferenceResults,
) -> Result<ConformanceDiff> {
    let mut details = Vec::new();
    let mut passed = true;

    // 1. Check decode token match
    let token_match = if let Some(expected_token) = fixture.steps.decode_1.selected_token_id {
        let matches = results.decode_token == expected_token;
        if !matches {
            details.push(format!(
                "Token mismatch: expected {}, got {}",
                expected_token, results.decode_token
            ));
            passed = false;
        }
        matches
    } else {
        details.push("No expected token in fixture".to_string());
        true // Don't fail if fixture doesn't specify expected token
    };

    // 2. Check top-K overlap
    let topk_overlap = analyze_topk_overlap(fixture, results)?;

    // Apply acceptance criteria: top-10 overlap ≥ 8/10 (80%)
    let min_overlap_ratio = 0.8;
    if topk_overlap.prefill_ratio < min_overlap_ratio {
        details.push(format!(
            "Prefill top-K overlap too low: {:.1}% < {:.1}%",
            topk_overlap.prefill_ratio * 100.0,
            min_overlap_ratio * 100.0
        ));
        passed = false;
    }

    if topk_overlap.decode_ratio < min_overlap_ratio {
        details.push(format!(
            "Decode top-K overlap too low: {:.1}% < {:.1}%",
            topk_overlap.decode_ratio * 100.0,
            min_overlap_ratio * 100.0
        ));
        passed = false;
    }

    // Add success details
    if passed {
        details.push("All conformance checks passed".to_string());
    }

    Ok(ConformanceDiff {
        passed,
        token_match,
        topk_overlap,
        details,
    })
}

/// Analyze top-K overlap between expected and actual results
fn analyze_topk_overlap(
    fixture: &ConformanceFixture,
    results: &InferenceResults,
) -> Result<TopKOverlap> {
    // Extract top-K from actual results
    let (prefill_tokens, _) = crate::conformance::fixtures::extract_topk_logits(
        &results.prefill_logits,
        fixture.steps.prefill_last.topk.k,
    )?;

    let (decode_tokens, _) = crate::conformance::fixtures::extract_topk_logits(
        &results.decode_logits,
        fixture.steps.decode_1.topk.k,
    )?;

    // Calculate overlaps
    let prefill_overlap =
        calculate_overlap(&fixture.steps.prefill_last.topk.token_ids, &prefill_tokens);

    let decode_overlap = calculate_overlap(&fixture.steps.decode_1.topk.token_ids, &decode_tokens);

    let prefill_total = fixture.steps.prefill_last.topk.k;
    let decode_total = fixture.steps.decode_1.topk.k;

    Ok(TopKOverlap {
        prefill_overlap,
        prefill_total,
        decode_overlap,
        decode_total,
        prefill_ratio: prefill_overlap as f32 / prefill_total as f32,
        decode_ratio: decode_overlap as f32 / decode_total as f32,
    })
}

/// Calculate overlap between two token ID lists
fn calculate_overlap(expected: &[usize], actual: &[usize]) -> usize {
    let expected_set: std::collections::HashSet<_> = expected.iter().collect();
    actual
        .iter()
        .filter(|token| expected_set.contains(token))
        .count()
}

/// Generate detailed diff report with first mismatch, max abs err, RMS
pub fn generate_diff_report(diff: &ConformanceDiff) -> String {
    let mut report = String::new();

    report.push_str("=== CONFORMANCE DIFF REPORT ===\n");
    report.push_str(&format!(
        "Overall Result: {}\n",
        if diff.passed { "PASS" } else { "FAIL" }
    ));
    report.push_str(&format!(
        "Token Match: {}\n",
        if diff.token_match { "✓" } else { "✗" }
    ));

    report.push_str("\nTop-K Overlap Analysis:\n");
    report.push_str(&format!(
        "  Prefill: {}/{} ({:.1}%)\n",
        diff.topk_overlap.prefill_overlap,
        diff.topk_overlap.prefill_total,
        diff.topk_overlap.prefill_ratio * 100.0
    ));
    report.push_str(&format!(
        "  Decode:  {}/{} ({:.1}%)\n",
        diff.topk_overlap.decode_overlap,
        diff.topk_overlap.decode_total,
        diff.topk_overlap.decode_ratio * 100.0
    ));

    report.push_str("\nDetails:\n");
    for detail in &diff.details {
        report.push_str(&format!("  - {}\n", detail));
    }

    report
}

/// Calculate maximum absolute error between two tensors
pub fn calculate_max_abs_error(expected: &Tensor, actual: &Tensor) -> Result<f32> {
    if expected.shape != actual.shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "max_abs_error".to_string(),
            expected: expected.shape.clone(),
            got: actual.shape.clone(),
        });
    }

    let max_error = expected
        .data
        .iter()
        .zip(actual.data.iter())
        .map(|(e, a)| (e - a).abs())
        .fold(0.0f32, |acc, err| acc.max(err));

    Ok(max_error)
}

/// Calculate RMS (Root Mean Square) error between two tensors
pub fn calculate_rms_error(expected: &Tensor, actual: &Tensor) -> Result<f32> {
    if expected.shape != actual.shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "rms_error".to_string(),
            expected: expected.shape.clone(),
            got: actual.shape.clone(),
        });
    }

    let sum_squared_error: f32 = expected
        .data
        .iter()
        .zip(actual.data.iter())
        .map(|(e, a)| (e - a).powi(2))
        .sum();

    let mean_squared_error = sum_squared_error / expected.data.len() as f32;
    Ok(mean_squared_error.sqrt())
}

/// Find first mismatch location between two tensors
pub fn find_first_mismatch(
    expected: &Tensor,
    actual: &Tensor,
    tolerance: f32,
) -> Result<Option<(usize, f32, f32)>> {
    if expected.shape != actual.shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "first_mismatch".to_string(),
            expected: expected.shape.clone(),
            got: actual.shape.clone(),
        });
    }

    for (i, (e, a)) in expected.data.iter().zip(actual.data.iter()).enumerate() {
        if (e - a).abs() > tolerance {
            return Ok(Some((i, *e, *a)));
        }
    }

    Ok(None)
}

/// Quick conformance check (returns true if passed)
pub fn check_conformance(fixture: &ConformanceFixture, results: &InferenceResults) -> Result<bool> {
    let diff = diff_conformance(fixture, results)?;
    Ok(diff.passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::fixtures::create_test_fixture;

    fn create_test_results() -> InferenceResults {
        InferenceResults {
            prefill_logits: Tensor::new(
                // Create logits where top tokens match fixture (0-9 are highest)
                (0..32000)
                    .map(|i| if i < 10 { 1.0 - i as f32 * 0.1 } else { -1.0 })
                    .collect(),
                vec![32000],
            )
            .unwrap(),
            decode_token: 1234, // Matches fixture
            decode_logits: Tensor::new(
                // Create logits where tokens 1230-1239 are highest
                (0..32000)
                    .map(|i| {
                        if (1230..1240).contains(&i) {
                            2.0 - (i - 1230) as f32 * 0.1
                        } else {
                            -1.0
                        }
                    })
                    .collect(),
                vec![32000],
            )
            .unwrap(),
        }
    }

    #[test]
    fn test_perfect_conformance() {
        let fixture = create_test_fixture();
        let results = create_test_results();

        let diff = diff_conformance(&fixture, &results).unwrap();

        assert!(diff.passed);
        assert!(diff.token_match);
        assert_eq!(diff.topk_overlap.prefill_overlap, 10); // Perfect overlap
        assert_eq!(diff.topk_overlap.decode_overlap, 10); // Perfect overlap
        assert_eq!(diff.topk_overlap.prefill_ratio, 1.0);
        assert_eq!(diff.topk_overlap.decode_ratio, 1.0);
    }

    #[test]
    fn test_token_mismatch() {
        let fixture = create_test_fixture();
        let mut results = create_test_results();
        results.decode_token = 9999; // Wrong token

        let diff = diff_conformance(&fixture, &results).unwrap();

        assert!(!diff.passed);
        assert!(!diff.token_match);
        assert!(diff.details.iter().any(|d| d.contains("Token mismatch")));
    }

    #[test]
    fn test_low_topk_overlap() {
        let fixture = create_test_fixture();
        let mut results = create_test_results();

        // Create logits with poor overlap (only first 2 tokens match)
        results.prefill_logits = Tensor::new(
            (0..32000)
                .map(|i| match i {
                    0 => 1.0,
                    1 => 0.9,
                    100..110 => 0.8 - (i - 100) as f32 * 0.1, // Different top tokens
                    _ => -1.0,
                })
                .collect(),
            vec![32000],
        )
        .unwrap();

        let diff = diff_conformance(&fixture, &results).unwrap();

        assert!(!diff.passed);
        assert!(diff.topk_overlap.prefill_ratio < 0.8); // Below threshold
        assert!(diff
            .details
            .iter()
            .any(|d| d.contains("Prefill top-K overlap too low")));
    }

    #[test]
    fn test_calculate_overlap() {
        let expected = vec![1, 2, 3, 4, 5];
        let actual = vec![1, 3, 5, 7, 9];

        let overlap = calculate_overlap(&expected, &actual);
        assert_eq!(overlap, 3); // 1, 3, 5 are common
    }

    #[test]
    fn test_diff_report_generation() {
        let fixture = create_test_fixture();
        let results = create_test_results();

        let diff = diff_conformance(&fixture, &results).unwrap();
        let report = generate_diff_report(&diff);

        assert!(report.contains("CONFORMANCE DIFF REPORT"));
        assert!(report.contains("PASS"));
        assert!(report.contains("Top-K Overlap Analysis"));
        assert!(report.contains("100.0%")); // Perfect overlap
    }

    #[test]
    fn test_max_abs_error() {
        let expected = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let actual = Tensor::new(vec![1.1, 1.9, 3.2], vec![3]).unwrap();

        let max_error = calculate_max_abs_error(&expected, &actual).unwrap();
        assert!((max_error - 0.2).abs() < 1e-6); // Max error is 0.2
    }

    #[test]
    fn test_rms_error() {
        let expected = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let actual = Tensor::new(vec![1.1, 1.9, 3.1], vec![3]).unwrap();

        let rms_error = calculate_rms_error(&expected, &actual).unwrap();
        // RMS = sqrt((0.1^2 + 0.1^2 + 0.1^2) / 3) = sqrt(0.03/3) = sqrt(0.01) = 0.1
        assert!((rms_error - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_first_mismatch() {
        let expected = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let actual = Tensor::new(vec![1.0, 2.1, 3.0, 4.0], vec![4]).unwrap();

        let mismatch = find_first_mismatch(&expected, &actual, 0.05).unwrap();
        assert_eq!(mismatch, Some((1, 2.0, 2.1))); // First mismatch at index 1

        // No mismatch with higher tolerance
        let no_mismatch = find_first_mismatch(&expected, &actual, 0.2).unwrap();
        assert_eq!(no_mismatch, None);
    }

    #[test]
    fn test_quick_conformance_check() {
        let fixture = create_test_fixture();
        let results = create_test_results();

        let passed = check_conformance(&fixture, &results).unwrap();
        assert!(passed);

        // Test with wrong token
        let mut bad_results = results;
        bad_results.decode_token = 9999;
        let passed = check_conformance(&fixture, &bad_results).unwrap();
        assert!(!passed);
    }
}
