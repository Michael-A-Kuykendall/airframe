pub mod client;

use async_trait::async_trait;
use anyhow::Result;

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn generate_response(&self, model_name: &str, prompt: &str) -> anyhow::Result<String>;
    async fn get_session_model(&self, session_id: &str) -> Option<String>;
    async fn list_models(&self) -> anyhow::Result<Vec<(String, serde_json::Value)>>;
    async fn set_session_model(&self, session_id: &str, model_name: &str) -> anyhow::Result<()>;
    async fn get_metrics(&self) -> anyhow::Result<serde_json::Value>;
    async fn generate_stream(&self, model_name: &str, prompt: &str, tx: tokio::sync::mpsc::Sender<String>) -> anyhow::Result<()>;
}
