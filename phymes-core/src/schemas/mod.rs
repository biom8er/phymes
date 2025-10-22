mod available_subjects;
pub use available_subjects::{create_schema_from_fields, AvailableSubjects, AvailableSubjectsTrait, create_values_record_batch, create_tools_record_batch, create_documents_batch};

mod blob;
pub use blob::{BlobSubject, create_blob_fields, create_blob_batch, BlobBuilderTraitExt};

mod mermaid;
pub use mermaid::{SessionMermaidSubject, create_session_mermaid_batch, create_mermaid_sequence_diagram_participants_template_batch};

mod chat;
pub use chat::{create_chat_record_batch, ChatTraitExt, ChatBuilderTraitExt};

mod queries;
pub use queries::{create_queries_batch, QueriesBuilderTraitExt};

mod user;
pub use user::{create_user_batch, create_user_session_contexts_batch, create_user_inbox_batch};

mod error;
pub use error::{create_error_batch, create_error_message_map_stream, create_error_message_map};

mod diagnostics;
pub use diagnostics::{DiagnosticsVisualizations, create_metrics_mermaid_gantt_batch, pivot_metrics_table, from_diagnostics_to_tables, get_metrics_as_gantt_table, get_metrics_as_mermaid_gantt};

mod session;
pub use session::{create_session_subjects_batch, create_session_subjects_num_rows_batch, create_session_tasks_batch, create_session_processors_batch, create_session_runtime_envs_batch};

// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
mod chat_completion;
pub use chat_completion::{ToolChoiceType, ChatCompletionRequest, ToolCall, Tool, ToolType};

mod common;
mod embedding;

// Based on openai-api-rs and modified to accomodate Apache Arrow
mod types;
pub use types::{Function, FunctionParameters, JSONSchemaType, JSONSchemaDefine};
