mod messages;
mod chat_builder;
mod tools;

// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
mod openai_chat_completion;
mod openai_common;

// Based on openai-api-rs and modified to accomodate Apache Arrow
mod chat_types;

pub use messages::{create_chat_fields, create_chat_record_batch};
pub use chat_builder::{ChatBuilderTraitExt, ChatTraitExt};
pub use chat_types::{Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType};
pub use openai_chat_completion::{
    ChatCompletionRequest, ChatCompletionResponse, FinishReason, Tool, ToolCall, ToolChoiceType,
    ToolType,
};
pub use tools::{
    create_bytes_fields, create_bytes_record_batch, create_route_bytes_fields,
    create_route_bytes_record_batch, create_tools_fields, create_tools_record_batch,
    create_values_fields, create_values_record_batch,
};
