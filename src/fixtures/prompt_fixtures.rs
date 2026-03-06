use crate::core::error::{LibshimmyError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Prompt fixture structure matching fixtures/prompts/<name>.tokens.json format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PromptFixture {
    pub prompt_name: String,
    pub prompt_text: String,
    pub token_ids: Vec<u32>,
    pub description: String,
}

impl PromptFixture {
    /// Validate that the fixture has required fields and valid data
    pub fn validate(&self) -> Result<()> {
        if self.prompt_name.is_empty() {
            return Err(LibshimmyError::FixtureError {
                msg: "Prompt name cannot be empty".to_string(),
            });
        }

        if self.token_ids.is_empty() {
            return Err(LibshimmyError::FixtureError {
                msg: format!("Prompt '{}' has no token IDs", self.prompt_name),
            });
        }

        // Validate token IDs are reasonable (not too large)
        for &token_id in &self.token_ids {
            if token_id > 100_000 {
                return Err(LibshimmyError::FixtureError {
                    msg: format!(
                        "Prompt '{}' contains suspicious token ID: {}",
                        self.prompt_name, token_id
                    ),
                });
            }
        }

        Ok(())
    }

    /// Get the number of tokens in this fixture
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }
}

/// Loader for prompt fixtures with validation and caching
pub struct PromptFixtureLoader {
    fixtures_dir: PathBuf,
    cache: HashMap<String, PromptFixture>,
    target_model_sha256: String,
}

impl PromptFixtureLoader {
    /// Create new fixture loader
    pub fn new<P: AsRef<Path>>(fixtures_dir: P, target_model_sha256: String) -> Self {
        Self {
            fixtures_dir: fixtures_dir.as_ref().to_path_buf(),
            cache: HashMap::new(),
            target_model_sha256,
        }
    }

    /// Create loader for V2 target model (TinyLlama Q4_0)
    pub fn v2_target<P: AsRef<Path>>(fixtures_dir: P) -> Self {
        Self::new(
            fixtures_dir,
            "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556".to_string(),
        )
    }

    /// Load a specific prompt fixture by name
    pub fn load_fixture(&mut self, prompt_name: &str) -> Result<&PromptFixture> {
        // Check cache first
        if self.cache.contains_key(prompt_name) {
            return Ok(self.cache.get(prompt_name).unwrap());
        }

        // Load from file
        let fixture_path = self
            .fixtures_dir
            .join("prompts")
            .join(format!("{}.tokens.json", prompt_name));

        if !fixture_path.exists() {
            return Err(LibshimmyError::FixtureError {
                msg: format!("Fixture file not found: {}", fixture_path.display()),
            });
        }

        let content =
            fs::read_to_string(&fixture_path).map_err(|e| LibshimmyError::FixtureError {
                msg: format!("Failed to read fixture {}: {}", fixture_path.display(), e),
            })?;

        let fixture: PromptFixture =
            serde_json::from_str(&content).map_err(|e| LibshimmyError::FixtureError {
                msg: format!("Failed to parse fixture {}: {}", fixture_path.display(), e),
            })?;

        // Validate fixture
        fixture.validate()?;

        // Validate prompt name matches filename
        if fixture.prompt_name != prompt_name {
            return Err(LibshimmyError::FixtureError {
                msg: format!(
                    "Fixture name mismatch: file '{}' contains prompt_name '{}'",
                    prompt_name, fixture.prompt_name
                ),
            });
        }

        // Cache and return
        self.cache.insert(prompt_name.to_string(), fixture);
        Ok(self.cache.get(prompt_name).unwrap())
    }

    /// Load all available fixtures in the prompts directory
    pub fn load_all_fixtures(&mut self) -> Result<Vec<PromptFixture>> {
        // First, get all fixture names
        let names = self.list_fixture_names()?;

        // Then load each fixture
        let mut fixtures = Vec::new();
        for name in names {
            let fixture = self.load_fixture(&name)?.clone();
            fixtures.push(fixture);
        }

        Ok(fixtures)
    }

    /// Get list of available fixture names
    pub fn list_fixture_names(&self) -> Result<Vec<String>> {
        let prompts_dir = self.fixtures_dir.join("prompts");

        if !prompts_dir.exists() {
            return Err(LibshimmyError::FixtureError {
                msg: format!("Prompts directory not found: {}", prompts_dir.display()),
            });
        }

        let mut names = Vec::new();

        for entry in fs::read_dir(&prompts_dir).map_err(|e| LibshimmyError::FixtureError {
            msg: format!("Failed to read prompts directory: {}", e),
        })? {
            let entry = entry.map_err(|e| LibshimmyError::FixtureError {
                msg: format!("Failed to read directory entry: {}", e),
            })?;

            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    // Remove .tokens suffix if present
                    let prompt_name = stem.strip_suffix(".tokens").unwrap_or(stem);
                    names.push(prompt_name.to_string());
                }
            }
        }

        names.sort();
        Ok(names)
    }

    /// Validate model SHA256 against target
    pub fn validate_model_sha256(&self, model_sha256: &str) -> Result<()> {
        if model_sha256 != self.target_model_sha256 {
            return Err(LibshimmyError::FixtureError {
                msg: format!(
                    "Model SHA256 mismatch: expected {}, got {}",
                    self.target_model_sha256, model_sha256
                ),
            });
        }
        Ok(())
    }

    /// Get the target model SHA256
    pub fn target_model_sha256(&self) -> &str {
        &self.target_model_sha256
    }

    /// Clear the fixture cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get number of cached fixtures
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_fixture(dir: &Path, name: &str, token_ids: Vec<u32>) -> Result<()> {
        let prompts_dir = dir.join("prompts");
        fs::create_dir_all(&prompts_dir)?;

        let fixture = PromptFixture {
            prompt_name: name.to_string(),
            prompt_text: format!("Test prompt {}", name),
            token_ids,
            description: format!("Test fixture for {}", name),
        };

        let fixture_path = prompts_dir.join(format!("{}.tokens.json", name));
        let content = serde_json::to_string_pretty(&fixture)?;
        fs::write(fixture_path, content)?;

        Ok(())
    }

    #[test]
    fn test_fixture_validation() {
        let valid_fixture = PromptFixture {
            prompt_name: "test".to_string(),
            prompt_text: "Hello".to_string(),
            token_ids: vec![1, 2, 3],
            description: "Test".to_string(),
        };
        assert!(valid_fixture.validate().is_ok());

        let empty_name = PromptFixture {
            prompt_name: "".to_string(),
            prompt_text: "Hello".to_string(),
            token_ids: vec![1, 2, 3],
            description: "Test".to_string(),
        };
        assert!(empty_name.validate().is_err());

        let empty_tokens = PromptFixture {
            prompt_name: "test".to_string(),
            prompt_text: "Hello".to_string(),
            token_ids: vec![],
            description: "Test".to_string(),
        };
        assert!(empty_tokens.validate().is_err());

        let suspicious_token = PromptFixture {
            prompt_name: "test".to_string(),
            prompt_text: "Hello".to_string(),
            token_ids: vec![1, 200_000], // Too large
            description: "Test".to_string(),
        };
        assert!(suspicious_token.validate().is_err());
    }

    #[test]
    fn test_fixture_loader() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        create_test_fixture(temp_dir.path(), "hello", vec![1, 2, 3])?;
        create_test_fixture(temp_dir.path(), "world", vec![4, 5, 6])?;

        let mut loader = PromptFixtureLoader::v2_target(temp_dir.path());

        // Test loading specific fixture
        let fixture = loader.load_fixture("hello")?;
        assert_eq!(fixture.prompt_name, "hello");
        assert_eq!(fixture.token_ids, vec![1, 2, 3]);

        // Test cache
        assert_eq!(loader.cache_size(), 1);

        // Test loading all fixtures
        let all_fixtures = loader.load_all_fixtures()?;
        assert_eq!(all_fixtures.len(), 2);

        // Test listing names
        let names = loader.list_fixture_names()?;
        assert_eq!(names, vec!["hello", "world"]);

        Ok(())
    }

    #[test]
    fn test_model_sha256_validation() {
        let temp_dir = TempDir::new().unwrap();
        let loader = PromptFixtureLoader::v2_target(temp_dir.path());

        // Should pass with correct SHA256
        assert!(loader
            .validate_model_sha256(
                "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"
            )
            .is_ok());

        // Should fail with incorrect SHA256
        assert!(loader.validate_model_sha256("wrong_sha256").is_err());
    }
}
