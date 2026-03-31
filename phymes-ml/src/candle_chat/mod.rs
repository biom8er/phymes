mod chat_config;
mod chat_processor;
mod message_parser_processor;
mod tool_call_config;
mod tool_call_processor;
mod tool_parser;

pub use chat_config::CandleChatConfig;
pub use chat_processor::{
    CandleChatProcessor, bench_chat_processor, process_logits_sampler, process_prompt_chat,
};
pub use message_parser_processor::MessageParserProcessor;
pub use tool_call_config::ToolCallConfig;
pub use tool_call_processor::ToolCallProcessor;
pub use tool_parser::extract_tool_calls_str;
