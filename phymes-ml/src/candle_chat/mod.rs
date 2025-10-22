mod chat_config;
mod chat_processor;
mod message_aggregator_processor;
mod message_parser_processor;
mod tool_parser;

pub use chat_config::CandleChatConfig;
pub use chat_processor::{
    CandleChatProcessor, bench_chat_processor, process_logits_sampler, process_prompt_chat,
};
pub use message_aggregator_processor::MessageAggregatorProcessor;
pub use message_parser_processor::MessageParserProcessor;
pub use tool_parser::extract_tool_calls_str;
