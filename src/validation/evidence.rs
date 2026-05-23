use crate::validation::errors::{SliceValidationError, ValidationResult};
use std::collections::HashMap;

/// Evidence checklist for slice validation
#[derive(Debug, Clone)]
pub struct EvidenceChecklist {
    items: HashMap<String, EvidenceItem>,
}

#[derive(Debug, Clone)]
pub struct EvidenceItem {
    pub name: String,
    pub value: Option<String>,
    pub required: bool,
}

impl EvidenceChecklist {
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
        }
    }

    /// Add required evidence item
    pub fn add_required(&mut self, name: &str) -> &mut Self {
        self.items.insert(
            name.to_string(),
            EvidenceItem {
                name: name.to_string(),
                value: None,
                required: true,
            },
        );
        self
    }

    /// Add optional evidence item
    pub fn add_optional(&mut self, name: &str) -> &mut Self {
        self.items.insert(
            name.to_string(),
            EvidenceItem {
                name: name.to_string(),
                value: None,
                required: false,
            },
        );
        self
    }

    /// Set evidence value
    pub fn set(&mut self, name: &str, value: String) -> &mut Self {
        if let Some(item) = self.items.get_mut(name) {
            item.value = Some(value);
        }
        self
    }

    /// Validate completeness
    pub fn validate(&self) -> ValidationResult<()> {
        let missing: Vec<String> = self
            .items
            .values()
            .filter(|item| item.required && item.value.is_none())
            .map(|item| item.name.clone())
            .collect();

        if !missing.is_empty() {
            return Err(SliceValidationError::EvidenceIncomplete {
                missing_items: missing,
            });
        }

        Ok(())
    }

    /// Print evidence checklist
    pub fn print(&self) {
        println!("=== EVIDENCE CHECKLIST ===");

        // Sort items for consistent output
        let mut sorted_items: Vec<_> = self.items.values().collect();
        sorted_items.sort_by(|a, b| a.name.cmp(&b.name));

        for item in sorted_items {
            let status = if let Some(ref value) = item.value {
                format!("✓ {}", value)
            } else if item.required {
                "✗ MISSING (REQUIRED)".to_string()
            } else {
                "- (optional)".to_string()
            };

            println!("  {}: {}", item.name, status);
        }
        println!("==========================");
    }

    /// Create standard V2 Slice 01 evidence checklist
    pub fn slice_01() -> Self {
        let mut checklist = Self::new();
        checklist
            .add_required("Model SHA256")
            .add_required("Model file size")
            .add_required("Prompt fixture identifier")
            .add_required("Token count expected vs produced")
            .add_required("Determinism confirmation")
            .add_required("Artifact path emitted");
        checklist
    }

    /// Create standard V2 Slice 02 evidence checklist
    pub fn slice_02() -> Self {
        let mut checklist = Self::new();
        checklist
            .add_required("Model SHA256")
            .add_required("KV cache growth validation")
            .add_required("Attention history usage")
            .add_required("Prefill/decode equivalence")
            .add_required("Artifact path emitted");
        checklist
    }

    /// Create standard V2 Slice 03 evidence checklist
    pub fn slice_03() -> Self {
        let mut checklist = Self::new();
        checklist
            .add_required("Model SHA256")
            .add_required("Oracle tool version")
            .add_required("Oracle command line")
            .add_required("Conformance result")
            .add_required("Artifact path emitted");
        checklist
    }
}

impl Default for EvidenceChecklist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── EvidenceChecklist construction ────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let c = EvidenceChecklist::new();
        assert!(c.validate().is_ok(), "empty checklist (no required items) should pass");
    }

    #[test]
    fn test_default_equals_new() {
        let a = EvidenceChecklist::new();
        let b = EvidenceChecklist::default();
        assert_eq!(a.items.len(), b.items.len());
    }

    // ── add_required / add_optional ───────────────────────────────────────────

    #[test]
    fn test_required_missing_fails_validation() {
        let mut c = EvidenceChecklist::new();
        c.add_required("sha256");
        let err = c.validate().expect_err("missing required item must fail");
        match err {
            SliceValidationError::EvidenceIncomplete { missing_items } => {
                assert!(missing_items.contains(&"sha256".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn test_optional_missing_does_not_fail() {
        let mut c = EvidenceChecklist::new();
        c.add_optional("extra");
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_required_set_passes() {
        let mut c = EvidenceChecklist::new();
        c.add_required("sha256");
        c.set("sha256", "abc123".to_string());
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_set_on_missing_key_is_noop() {
        // set() on a key that was never added should not panic or create an entry
        let mut c = EvidenceChecklist::new();
        c.set("nonexistent", "value".to_string());
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_multiple_required_all_must_be_set() {
        let mut c = EvidenceChecklist::new();
        c.add_required("a");
        c.add_required("b");
        c.add_required("c");
        c.set("a", "1".to_string());
        c.set("b", "2".to_string());
        // "c" not set
        assert!(c.validate().is_err());
        c.set("c", "3".to_string());
        assert!(c.validate().is_ok());
    }

    // ── Slice factory constructors ────────────────────────────────────────────

    #[test]
    fn test_slice_01_has_required_fields() {
        let mut c = EvidenceChecklist::slice_01();
        // Should fail until all 6 required items are filled
        assert!(c.validate().is_err());

        c.set("Model SHA256", "x".to_string());
        c.set("Model file size", "y".to_string());
        c.set("Prompt fixture identifier", "p1".to_string());
        c.set("Token count expected vs produced", "16 expected, 16 produced".to_string());
        c.set("Determinism confirmation", "PASS".to_string());
        c.set("Artifact path emitted", "/tmp/foo.json".to_string());

        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_slice_02_has_required_fields() {
        let mut c = EvidenceChecklist::slice_02();
        assert!(c.validate().is_err());

        c.set("Model SHA256", "x".to_string());
        c.set("KV cache growth validation", "PASS".to_string());
        c.set("Attention history usage", "PASS".to_string());
        c.set("Prefill/decode equivalence", "PASS".to_string());
        c.set("Artifact path emitted", "/tmp/foo.json".to_string());

        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_slice_03_has_required_fields() {
        let mut c = EvidenceChecklist::slice_03();
        assert!(c.validate().is_err());

        c.set("Model SHA256", "x".to_string());
        c.set("Oracle tool version", "v1.0".to_string());
        c.set("Oracle command line", "cmd".to_string());
        c.set("Conformance result", "PASS".to_string());
        c.set("Artifact path emitted", "/tmp/foo.json".to_string());

        assert!(c.validate().is_ok());
    }

    // ── EvidenceItem fields ───────────────────────────────────────────────────

    #[test]
    fn test_required_flag_is_true_for_add_required() {
        let mut c = EvidenceChecklist::new();
        c.add_required("must-have");
        let item = &c.items["must-have"];
        assert!(item.required);
        assert!(item.value.is_none());
    }

    #[test]
    fn test_required_flag_is_false_for_add_optional() {
        let mut c = EvidenceChecklist::new();
        c.add_optional("nice-to-have");
        let item = &c.items["nice-to-have"];
        assert!(!item.required);
    }

    // ── print() exercises the display path ───────────────────────────────────

    #[test]
    fn test_print_does_not_panic() {
        let mut c = EvidenceChecklist::slice_01();
        c.set("Model SHA256", "deadbeef".to_string());
        // Calling print() should not panic regardless of fill state
        c.print();
    }
}
