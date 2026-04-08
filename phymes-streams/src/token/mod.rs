mod candle_embed_stream;
mod chat_builder;
mod candle_chat_stream;
mod message_parser_stream;
#[cfg(feature = "api")]
mod openai_chat_stream;
#[cfg(feature = "api")]
mod openai_embed_stream;
mod tool_call_config;
mod tool_call_stream;
mod tool_parser;

pub use candle_embed_stream::CandleEmbedStream;
pub use chat_builder::{ChatBuilderTraitExt, ChatTraitExt};
pub use candle_chat_stream::CandleChatStream;
pub use message_parser_stream::MessageParserStream;
#[cfg(feature = "api")]
pub use openai_chat_stream::OpenAIChatStream;
#[cfg(feature = "api")]
pub use openai_embed_stream::OpenAIEmbedStream;
pub use tool_call_config::ToolCallConfig;
pub use tool_call_stream::ToolCallStream;
pub use tool_parser::extract_tool_calls_str;
