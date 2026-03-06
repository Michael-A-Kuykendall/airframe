pub mod local_adapter;
pub mod shimmy_adapter;
pub mod http_adapter;
pub mod mock_adapter;
pub mod ws_adapter;

pub use local_adapter::LocalInferenceAdapter;
pub use shimmy_adapter::ShimmyServerAdapter;
pub use http_adapter::HttpInferenceAdapter;
pub use mock_adapter::MockInferenceAdapter;
pub use ws_adapter::WsInferenceAdapter;
