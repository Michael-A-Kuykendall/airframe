use anyhow::Result;

pub struct RenderContext {
    pub session_id: String,
    pub max_tokens: usize,
}

impl Default for RenderContext {
    fn default() -> Self {
        Self {
            session_id: "default".to_string(),
            max_tokens: 2048,
        }
    }
}
