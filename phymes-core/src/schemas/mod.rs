pub mod blob;
pub mod mermaid;
pub mod chat;
pub mod queries;
pub mod user;
pub mod available_subjects;
pub mod error;
pub mod diagnostics;

// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
pub mod chat_completion;
pub mod common;
pub mod embedding;

// Based on openai-api-rs and modified to accomodate Apache Arrow
pub mod types;
