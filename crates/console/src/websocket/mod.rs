pub mod client;

use async_trait::async_trait;
use anyhow::Result;

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn generate_response(&mut self, prompt: &str) -> std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<String>> + Send>>;
    async fn get_session_model(&self) -> Result<String>;
    async fn list_models(&self) -> Result<Vec<String>>;
    async fn set_session_model(&mut self, model: &str) -> Result<()>;
    async fn get_metrics(&self) -> Result<String>;
    async fn generate_stream(&mut self, prompt: &str) -> Result<()>;
}
