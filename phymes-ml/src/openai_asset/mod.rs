#[cfg(feature = "openai_api")]
use reqwest::Response;
#[cfg(feature = "openai_api")]
use std::{future::Future, pin::Pin};

/// The state of the OpenAI API request
///
/// We need to capture each stage of the request so that
///   the connection is not dropped during repeated polling
///   of the stream.
#[cfg(feature = "openai_api")]
pub enum OpenAIRequestState {
    NotStarted,
    Connecting(Pin<Box<dyn Future<Output = Result<Response, reqwest::Error>> + Send + 'static>>),
    ToText(Pin<Box<dyn Future<Output = Result<String, reqwest::Error>> + Send + 'static>>),
    Done,
}

mod available_openai_assets;
pub use available_openai_assets::AvailableOpenAIAssets;
