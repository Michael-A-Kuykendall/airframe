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
