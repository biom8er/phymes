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
enum OpenAIRequestState {
    NotStarted,
    Connecting(Pin<Box<dyn Future<Output = Result<Response, reqwest::Error>> + Send + 'static>>),
    ToText(Pin<Box<dyn Future<Output = Result<String, reqwest::Error>> + Send + 'static>>),
    Done,
}

#[cfg(feature = "openai_api")]
pub mod chat_processor;
#[cfg(feature = "openai_api")]
pub mod embed_processor;
pub mod openai_which;

// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
pub mod chat_completion;
pub mod common;
pub mod embedding;
pub mod types;
