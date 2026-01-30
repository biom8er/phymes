mod available_subjects;
pub use available_subjects::{
    AvailableSubjects, AvailableSubjectsTrait, create_documents_batch, create_schema_from_fields,
    create_tools_record_batch, create_values_record_batch,
};

mod blob;
pub use blob::{BlobBuilderTraitExt, BlobSubject, create_blob_batch, create_blob_fields};

mod mermaid;
pub use mermaid::{
    SessionMermaidSubject, create_mermaid_content_template_batch,
    create_mermaid_sequence_diagram_participants_template_batch, create_session_mermaid_batch,
};

mod chat;
pub use chat::{ChatBuilderTraitExt, ChatTraitExt, create_chat_fields, create_chat_record_batch};

mod queries;
pub use queries::create_queries_batch;

mod user;
pub use user::{
    JoinUserInboxSessionContextsMermaidDiagrams, UserSubject, create_user_batch,
    create_user_inbox_batch, create_user_session_contexts_batch,
};

mod error;
pub use error::{create_error_message_map, create_error_message_map_stream};

mod diagnostics;
pub use diagnostics::{
    DiagnosticsVisualizations, create_metrics_mermaid_gantt_batch, from_diagnostics_to_tables,
};

mod session;
pub use session::{
    create_session_processors_batch, create_session_runtime_envs_batch,
    create_session_subjects_batch, create_session_supersteps_batch, create_session_tasks_batch,
    create_session_tasks_check_batch, create_session_tasks_publish_batch,
    create_session_tasks_run_log_batch, create_session_tasks_subscribe_aggregate_batch,
    create_session_tasks_subscribe_batch, create_session_tasks_subscribe_publish_batch,
};

mod subjects;
pub use subjects::{create_subjects_change_log_batch, create_subjects_num_rows_batch};

mod graph;
pub use graph::{
    create_n_quads_batch, create_n_triples_batch, create_parse_n_quads_batch,
    create_parse_owl_batch, create_parse_xml_batch,
};

// Based on openai-api-rs <https://github.com/dongri/openai-api-rs>
mod chat_completion;
pub use chat_completion::{
    ChatCompletionRequest, ChatCompletionResponse, FinishReason, Tool, ToolCall, ToolChoiceType,
    ToolType,
};

mod common;

mod embedding;
pub use embedding::{EmbeddingRequest, EmbeddingResponse, EncodingFormat};

// Based on openai-api-rs and modified to accomodate Apache Arrow
mod chat_types;
pub use chat_types::{Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType};
