use crate::websocket::InferenceBackend;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

pub struct ShimmyServerAdapter {
    base_url: String,
    client: Client,
}

impl ShimmyServerAdapter {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl InferenceBackend for ShimmyServerAdapter {
    async fn generate_response(&self, _model_name: &str, prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/api/repro/inference", self.base_url);
        let body = serde_json::json!({
            "task": "chat",
            "prompt": prompt,
            "prompt_mode": "raw",
            "max_tokens": 1024
        });

        let resp = self.client.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let error_text = resp.text().await?;
            return Err(anyhow::anyhow!("Submit error: {}", error_text));
        }

        let submit_result: Value = resp.json().await?;
        let job_id = submit_result["job_id"].as_str().ok_or_else(|| anyhow::anyhow!("Missing job_id"))?;

        let status_url = format!("{}/api/repro/job-status?job_id={}", self.base_url, job_id);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let st_resp = self.client.get(&status_url).send().await?;
            if !st_resp.status().is_success() {
                continue;
            }
            let st_json: Value = st_resp.json().await?;
            let status = st_json["status"].as_str().unwrap_or("");
            if status == "Completed" || status == "Failed" {
                if status == "Failed" {
                    return Err(anyhow::anyhow!("Job failed"));
                }
                if let Some(text) = st_json["result"]["text"].as_str() {
                    return Ok(text.to_string());
                } else {
                    return Ok("".to_string());
                }
            }
        }
    }

    async fn get_session_model(&self, _session_id: &str) -> Option<String> { None }
    async fn set_session_model(&self, _session_id: &str, _model_name: &str) -> anyhow::Result<()> { Ok(()) }
    async fn list_models(&self) -> anyhow::Result<Vec<(String, Value)>> { Ok(vec![("shimmy-gpu".to_string(), serde_json::json!({}))]) }
    async fn get_metrics(&self) -> anyhow::Result<Value> { Ok(serde_json::json!({})) }
    async fn generate_stream(&self, _model: &str, _prompt: &str, _tx: tokio::sync::mpsc::Sender<String>) -> anyhow::Result<()> { Err(anyhow::anyhow!("Streaming not supported yet")) }
}
