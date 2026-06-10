use crate::websocket::InferenceBackend;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use airframe::core::model::Model;
use airframe::family::llama::LlamaModel;
use airframe::runtime::{engine::Engine, sampling::Sampler};
use shimmytok::Tokenizer;
use std::path::Path;

/// Local inference adapter — runs airframe engine directly, no server needed.
/// This is the single-command launch path: no shimmy serve, no second terminal.
#[derive(Clone)]
pub struct LocalInferenceAdapter {
    model_path: String,
    state: Arc<Mutex<Option<(Model, Tokenizer)>>>,
}

impl LocalInferenceAdapter {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            state: Arc::new(Mutex::new(None)),
        }
    }

    /// Model display name derived from the file stem
    pub fn model_name(&self) -> String {
        Path::new(&self.model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("local-model")
            .to_string()
    }

    /// Load model on first use, reuse on subsequent calls
    async fn ensure_loaded(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock().await;
        if state.is_none() {
            println!("🔧 Loading model: {}", self.model_name());
            let tokenizer = Tokenizer::from_gguf_file(&self.model_path)?;
            // Use generic GGUF loader — works for any model, not just TinyLlama
            let model = Model::from_gguf(Path::new(&self.model_path))?;
            println!("✅ Model loaded: {}", self.model_name());
            *state = Some((model, tokenizer));
        }
        Ok(())
    }
}

#[async_trait]
impl InferenceBackend for LocalInferenceAdapter {
    async fn list_models(&self) -> anyhow::Result<Vec<(String, serde_json::Value)>> {
        Ok(vec![(
            self.model_name(),
            json!({
                "provider": "airframe-local",
                "type": "local",
                "path": self.model_path
            })
        )])
    }

    async fn generate_response(&self, _model_name: &str, prompt: &str) -> anyhow::Result<String> {
        self.ensure_loaded().await?;
        let state = self.state.lock().await;
        let (model, tokenizer) = state.as_ref().unwrap();

        let tokens = tokenizer.encode(prompt, true)?;
        let llama_model = LlamaModel::from_spec(model.spec.clone());
        let mut engine = Engine::new(llama_model);
        let sampler = Sampler::greedy();

        let prompt_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        let mut logits = engine.prefill(&prompt_ids, &model.weights)?;

        let mut output = String::new();
        for _ in 0..512 {
            let next_token = sampler.sample(&logits)?;
            if next_token as u32 == tokenizer.eos_token() {
                break;
            }
            if let Ok(text) = tokenizer.decode(&[next_token as u32], true) {
                output.push_str(&text);
            }
            logits = engine.decode(next_token, &model.weights)?;
        }

        Ok(output)
    }

    async fn get_session_model(&self, _session_id: &str) -> Option<String> {
        Some(self.model_name())
    }

    async fn set_session_model(&self, _session_id: &str, _model_name: &str) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_metrics(&self) -> anyhow::Result<serde_json::Value> {
        Ok(json!({
            "backend": "airframe-local",
            "model": self.model_name(),
            "path": self.model_path
        }))
    }

    async fn generate_stream(
        &self,
        _model_name: &str,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        self.ensure_loaded().await?;
        let state = self.state.lock().await;
        let (model, tokenizer) = state.as_ref().unwrap();

        let tokens = tokenizer.encode(prompt, true)?;
        let llama_model = LlamaModel::from_spec(model.spec.clone());
        let mut engine = Engine::new(llama_model);
        let sampler = Sampler::greedy();

        let prompt_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        let mut logits = engine.prefill(&prompt_ids, &model.weights)?;

        for _ in 0..512 {
            let next_token = sampler.sample(&logits)?;
            if next_token as u32 == tokenizer.eos_token() {
                break;
            }
            if let Ok(text) = tokenizer.decode(&[next_token as u32], true) {
                if tx.send(text).await.is_err() {
                    break; // Receiver dropped — user quit
                }
            }
            logits = engine.decode(next_token, &model.weights)?;
        }

        Ok(())
    }
}
