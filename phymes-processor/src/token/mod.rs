mod candle_chat_processor;
mod candle_embed_processor;
mod chat_builder;
mod message_parser_processor;
#[cfg(feature = "api")]
mod openai_chat_processor;
#[cfg(feature = "api")]
mod openai_embed_processor;
mod token_service;
mod tool_call_processor;

pub use candle_chat_processor::CandleChatProcessor;
pub use candle_embed_processor::CandleEmbedProcessor;
pub use message_parser_processor::MessageParserProcessor;
#[cfg(feature = "api")]
pub use openai_chat_processor::OpenAIChatProcessor;
#[cfg(feature = "api")]
pub use openai_embed_processor::OpenAIEmbedProcessor;
pub use token_service::TokenStreamTraitExt;
pub use tool_call_processor::ToolCallProcessor;