use crate::{schemas::{blob::create_blob_fields, chat::create_chat_fields, error::create_error_fields, mermaid::create_mermaid_fields, metrics::{create_metrics_fields, create_metrics_mermaid_gantt_fields}, queries::create_queries_fields, user::{create_join_user_inbox_session_contexts_fields, create_join_user_inbox_session_contexts_mermaid_diagrams_fields, create_user_fields, create_user_inbox_fields, create_user_session_contexts_fields}}, session::common_traits::{BuildableTrait, BuilderTrait}, table::table_trait::{Table, TableBuilder, TableBuilderTrait}};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, StringArray},
    datatypes::{DataType, Field, Fields, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use clap::ValueEnum;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{fmt::Display, sync::Arc};

/// Generate a timestamp that can be added to the message table
pub fn create_timestamp_str() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.format("%a %b %e %T %Y").to_string()
}

/// Generate a timestamp that can be added to the message table
pub fn create_timestamp_micros() -> i64 {
    let now: DateTime<Utc> = Utc::now();
    now.timestamp_micros()
}

/// Convert timestamp in micro seconds to a formatted string
pub fn convert_timestamp_micros_to_str(timestamp_micros: i64) -> String {
    // Convert microseconds to seconds and nanoseconds
    let datetime = DateTime::from_timestamp(
        timestamp_micros / 1_000_000,                    // seconds
        ((timestamp_micros % 1_000_000) * 1_000) as u32, // nanoseconds
    )
    .unwrap();

    // Format as a string
    datetime.format("%a %b %e %T %Y").to_string()
}

pub fn create_schema_from_fields(f: &dyn Fn() -> Fields) -> SchemaRef {
    Arc::new(Schema::new(f()))
}

pub fn create_values_fields() -> Fields {
    let field_names = ["name", "publisher", "subject", "values"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_values_record_batch(
    names: Vec<String>,
    publishers: Vec<String>,
    subjects: Vec<String>,
    values: Vec<String>,
) -> Result<RecordBatch> {
    let names: ArrayRef = Arc::new(StringArray::from(names));
    let publishers: ArrayRef = Arc::new(StringArray::from(publishers));
    let subjects: ArrayRef = Arc::new(StringArray::from(subjects));
    let values: ArrayRef = Arc::new(StringArray::from(values));
    let batch = RecordBatch::try_from_iter(vec![
        ("name", names),
        ("publisher", publishers),
        ("subject", subjects),
        ("values", values),
    ])?;
    Ok(batch)
}

pub fn create_config_fields() -> Fields {
    let field_names = ["values"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_tools_fields() -> Fields {
    let field_names = ["tool_id", "tool"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_tools_record_batch(
    tool_ids: Vec<String>,
    tools: Vec<String>,
) -> Result<RecordBatch> {
    let tool_ids: ArrayRef = Arc::new(StringArray::from(tool_ids));
    let tools: ArrayRef = Arc::new(StringArray::from(tools));
    let batch = RecordBatch::try_from_iter(vec![
        ("tool_id", tool_ids),
        ("tool", tools),
    ])?;
    Ok(batch)
}

pub fn create_documents_fields() -> Fields {
    let field_names = ["chunk_id", "document_id", "text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_documents_batch(
    chunk_id: Vec<String>,
    document_id: Vec<String>,
    text: Vec<String>,
) -> Result<RecordBatch> {
    let chunk_id: ArrayRef = Arc::new(StringArray::from(chunk_id));
    let document_id: ArrayRef = Arc::new(StringArray::from(document_id));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let batch = RecordBatch::try_from_iter(vec![
        ("chunk_id", chunk_id),
        ("document_id", document_id),
        ("text", text),
    ])?;
    Ok(batch)
}

pub fn create_document_embeddings_fields() -> Fields {
    let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
    let document_id = Field::new("document_id", DataType::Utf8, false);
    let list_data_type = DataType::List(
        Arc::new(Field::new_list_field(DataType::Float32, false))
    );
    let embedding = Field::new("embedding", list_data_type, false);
    Fields::from(vec![chunk_id, document_id, embedding])
}

pub fn create_query_embeddings_fields() -> Fields {
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let list_data_type = DataType::List(
        Arc::new(Field::new_list_field(DataType::Float32, false))
    );
    let embedding = Field::new("embedding", list_data_type, false);
    Fields::from(vec![query_id, embedding])
}

pub fn create_embeddings_scores_fields() -> Fields {
    let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let score = Field::new("score", DataType::Float32, false);
    Fields::from(vec![chunk_id, query_id, score])
}

pub fn create_join_chunks_scores_fields() -> Fields {
    let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let score = Field::new("score", DataType::Float32, false);
    let document_id = Field::new("document_id", DataType::Utf8, false);
    let text = Field::new("text", DataType::Utf8, false);
    Fields::from(vec![
        chunk_id,
        query_id,
        score,
        document_id,
        text,
    ])
}

pub trait AvailableSubjectsTrait {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Table>;
    fn to_table_builder(&self, name: Option<&str>) -> TableBuilder;
    fn to_schema(&self) -> SchemaRef;
}

/// The available subject schmeas
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableSubjects {
    #[value(name = "Messages")]
    Messages,
    #[default]
    #[value(name = "Values")]
    Values,
    #[value(name = "Configs")]
    Configs,
    #[value(name = "Tools")]
    Tools,
    #[value(name = "Documents")]
    Documents,
    #[value(name = "Queries")]
    Queries,
    #[value(name = "DocumentEmbeddings")]
    DocumentEmbeddings,
    #[value(name = "QueryEmbeddings")]
    QueryEmbeddings,
    #[value(name = "EmbeddingScores")]
    EmbeddingScores,
    #[value(name = "JoinChunksScores")]
    JoinChunksScores,
    #[value(name = "Blob")]
    Blob,
    #[value(name = "User")]
    User,
    #[value(name = "UserSessionContexts")]
    UserSessionContexts,
    #[value(name = "UserInbox")]
    UserInbox,
    #[value(name = "JoinUserInboxSessionContexts")]
    JoinUserInboxSessionContexts,
    #[value(name = "JoinUserInboxSessionContextsMermaid")]
    JoinUserInboxSessionContextsMermaid,
    #[value(name = "Mermaid")]
    Mermaid,
    #[value(name = "Errors")]
    Errors,
    #[value(name = "Metrics")]
    Metrics,
    #[value(name = "MetricsMermaidGantt")]
    MetricsMermaidGantt,
}

impl Display for AvailableSubjects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AvailableSubjects::Messages => write!(f, "Messages"),
            AvailableSubjects::Values => write!(f, "Values"),
            AvailableSubjects::Configs => write!(f, "Configs"),
            AvailableSubjects::Tools => write!(f, "Tools"),
            AvailableSubjects::Documents => write!(f, "Documents"),
            AvailableSubjects::Queries => write!(f, "Queries"),
            AvailableSubjects::DocumentEmbeddings => write!(f, "DocumentEmbeddings"),
            AvailableSubjects::QueryEmbeddings => write!(f, "QueryEmbeddings"),
            AvailableSubjects::EmbeddingScores => write!(f, "EmbeddingScores"),
            AvailableSubjects::JoinChunksScores => write!(f, "JoinChunksScores"),
            AvailableSubjects::Blob => write!(f, "Blob"),
            AvailableSubjects::User => write!(f, "User"),
            AvailableSubjects::UserSessionContexts => write!(f, "UserSessionContexts"),
            AvailableSubjects::UserInbox => write!(f, "UserInbox"),
            AvailableSubjects::JoinUserInboxSessionContexts => write!(f, "JoinUserInboxSessionContexts"),
            AvailableSubjects::JoinUserInboxSessionContextsMermaid => write!(f, "JoinUserInboxSessionContextsMermaid"),
            AvailableSubjects::Mermaid => write!(f, "Mermaid"),
            AvailableSubjects::Errors => write!(f, "Errors"),
            AvailableSubjects::Metrics => write!(f, "Metrics"),
            AvailableSubjects::MetricsMermaidGantt => write!(f, "MetricsMermaidGantt"),
        }
    }
}

impl AvailableSubjectsTrait for AvailableSubjects {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Table> {
        let builder = self.to_table_builder(name);
        let batches = batches.unwrap_or_default();
        builder.with_record_batches(batches)?.build()
    }
    fn to_table_builder(&self, name: Option<&str>) -> TableBuilder {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        Table::get_builder().with_name(&name).with_schema(self.to_schema())
    }
    fn to_schema(&self) -> SchemaRef {
        match self {
            AvailableSubjects::Messages => create_schema_from_fields(&create_chat_fields),
            AvailableSubjects::Values => create_schema_from_fields(&create_values_fields),
            AvailableSubjects::Configs => create_schema_from_fields(&create_config_fields),
            AvailableSubjects::Tools => create_schema_from_fields(&create_tools_fields),
            AvailableSubjects::Documents => {
                create_schema_from_fields(&create_documents_fields)
            }
            AvailableSubjects::Queries => create_schema_from_fields(&create_queries_fields),
            AvailableSubjects::DocumentEmbeddings => {
                create_schema_from_fields(&create_document_embeddings_fields)
            }
            AvailableSubjects::QueryEmbeddings => {
                create_schema_from_fields(&create_query_embeddings_fields)
            }
            AvailableSubjects::EmbeddingScores => {
                create_schema_from_fields(&create_embeddings_scores_fields)
            }
            AvailableSubjects::JoinChunksScores => {
                create_schema_from_fields(&create_join_chunks_scores_fields)
            }
            AvailableSubjects::Blob => create_schema_from_fields(&create_blob_fields),
            AvailableSubjects::User => create_schema_from_fields(&create_user_fields),
            AvailableSubjects::UserSessionContexts => create_schema_from_fields(&create_user_session_contexts_fields),
            AvailableSubjects::UserInbox => create_schema_from_fields(&create_user_inbox_fields),
            AvailableSubjects::JoinUserInboxSessionContexts => create_schema_from_fields(&create_join_user_inbox_session_contexts_fields),
            AvailableSubjects::JoinUserInboxSessionContextsMermaid => create_schema_from_fields(&create_join_user_inbox_session_contexts_mermaid_diagrams_fields),
            AvailableSubjects::Mermaid => create_schema_from_fields(&create_mermaid_fields),
            AvailableSubjects::Errors => create_schema_from_fields(&create_error_fields),
            AvailableSubjects::Metrics => create_schema_from_fields(&create_metrics_fields),
            AvailableSubjects::MetricsMermaidGantt => create_schema_from_fields(&create_metrics_mermaid_gantt_fields),
        }
    }
}