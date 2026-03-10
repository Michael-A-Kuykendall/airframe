use crate::websocket::InferenceBackend;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;

#[derive(Clone)]
pub struct ShimmyServerAdapter {
    base_url: String,
    session_id: String,
    client: Client,
}

impl ShimmyServerAdapter {
    pub fn new(base_url: String, session_id: String) -> Self {
        Self {
            base_url,
            session_id,
            client: Client::new(),
        }
    }

    fn request_body(&self, prompt: &str, stream: bool) -> Value {
        serde_json::json!({
            "task": "chat",
            "prompt": prompt,
            "prompt_mode": "raw",
            "max_tokens": 1024,
            "session_id": self.session_id,
            "stream": stream
        })
    }
}

#[async_trait]
impl InferenceBackend for ShimmyServerAdapter {
    async fn generate_response(&self, _model_name: &str, prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/api/repro/inference", self.base_url);
        let body = self.request_body(prompt, false);

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
            let status = st_json["status"].as_str().unwrap_or("").to_lowercase();
            if status == "completed" || status == "failed" {
                if status == "failed" {
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
    async fn generate_stream(&self, _model: &str, prompt: &str, tx: tokio::sync::mpsc::Sender<String>) -> anyhow::Result<()> {
        let url = format!("{}/api/repro/inference", self.base_url);
        let body = self.request_body(prompt, true);

        let resp = self.client.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let error_text = resp.text().await?;
            return Err(anyhow::anyhow!("Submit error: {}", error_text));
        }

        let submit_result: Value = resp.json().await?;
        let job_id = submit_result["job_id"].as_str().ok_or_else(|| anyhow::anyhow!("Missing job_id"))?;

        let stream_url = format!("{}/api/repro/job-stream?job_id={}", self.base_url, job_id);
        let stream_resp = self.client.get(&stream_url).send().await?;
        if !stream_resp.status().is_success() {
            let error_text = stream_resp.text().await?;
            return Err(anyhow::anyhow!("Stream error: {}", error_text));
        }

        let mut byte_stream = stream_resp.bytes_stream();
        while let Some(item) = byte_stream.next().await {
            let bytes = item?;
            if bytes.is_empty() {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes).to_string();
            if tx.send(text).await.is_err() {
                break;
            }
        }

        Ok(())
    }
}
