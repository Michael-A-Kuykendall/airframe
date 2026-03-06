//! Adapter abstraction for embedding projections.
//!
//! Ensures projection prompts are not leaked into the runtime prompt stream.

#[derive(Debug, Clone)]
pub struct Adapter {
    pub id: String,
    /// Encoded projection that the adapter represents (if any).
    pub encoded_projection: Option<String>,
}

impl Adapter {
    pub fn new(id: &str, encoded_projection: Option<&str>) -> Self {
        Self {
            id: id.to_string(),
            encoded_projection: encoded_projection.map(|s| s.to_string()),
        }
    }

    /// Apply adapter to a prompt, stripping any encoded projection text.
    pub fn apply_to_prompt(&self, base_prompt: &str) -> String {
        if let Some(ref proj) = self.encoded_projection {
            // Remove occurrences of the explicit projection text if present
            base_prompt.replace(proj, "")
        } else {
            base_prompt.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_mode_no_projection_prompt() {
        let proj = "#Projection: keep this secret";
        let base = format!("User wants X. {} More text.", proj);

        let adapter = Adapter::new("a1", Some(proj));
        let applied = adapter.apply_to_prompt(&base);
        assert!(
            !applied.contains(proj),
            "Adapter application must not include explicit projection prompt"
        );
    }

    #[test]
    fn test_adapter_without_projection_preserves_prompt() {
        let base = "Just a normal prompt";
        let adapter = Adapter::new("a2", None);
        assert_eq!(adapter.apply_to_prompt(base), base);
    }
}
