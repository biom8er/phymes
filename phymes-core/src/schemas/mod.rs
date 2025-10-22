mod available_subjects;
pub use available_subjects::{
    AvailableSubjects, AvailableSubjectsTrait, create_documents_batch, create_schema_from_fields,
    create_tools_record_batch, create_values_record_batch,
};

mod blob;
pub use blob::{BlobBuilderTraitExt, BlobSubject, create_blob_batch, create_blob_fields};

mod mermaid;
pub use mermaid::{
    SessionMermaidSubject, create_mermaid_sequence_diagram_participants_template_batch,
    create_session_mermaid_batch,
};

mod chat;
pub use chat::{ChatBuilderTraitExt, ChatTraitExt, create_chat_fields, create_chat_record_batch};

mod queries;
pub use queries::{QueriesBuilderTraitExt, create_queries_batch};

mod user;
pub use user::{
    JoinUserInboxSessionContextsMermaidDiagrams, UserSubject, create_user_batch,
    create_user_inbox_batch, create_user_session_contexts_batch,
};

mod error;
pub use error::{create_error_batch, create_error_message_map, create_error_message_map_stream};

mod diagnostics;
pub use diagnostics::{
    DiagnosticsVisualizations, create_metrics_mermaid_gantt_batch, from_diagnostics_to_tables,
    get_metrics_as_gantt_table, get_metrics_as_mermaid_gantt, pivot_metrics_table,
};

mod session;
pub use session::{
    create_session_processors_batch, create_session_runtime_envs_batch,
    create_session_subjects_batch, create_session_subjects_num_rows_batch,
    create_session_tasks_batch,
};

// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
mod chat_completion;
pub use chat_completion::{ChatCompletionRequest, Tool, ToolCall, ToolChoiceType, ToolType};

mod common;
mod embedding;

// Based on openai-api-rs and modified to accomodate Apache Arrow
mod types;
pub use types::{Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType};
