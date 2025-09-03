use crate::{session::common_traits::BuilderTrait, table::arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait}};

use anyhow::{anyhow, Result};
use arrow::{
    array::{ArrayRef, Int64Array, ListBuilder, StringArray, UInt8Builder},
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

pub fn create_table_from_fields(
    name: &str,
    batches: Option<Vec<RecordBatch>>,
    f: &dyn Fn() -> Fields,
) -> Result<ArrowTable> {
    ArrowTableBuilder::new()
        .with_name(name)
        .with_schema(create_schema_from_fields(f))
        .with_record_batches(batches.unwrap_or(Vec::new()))?
        .build()
}

pub fn create_table_from_fields_and_struct<T>(
    name: &str,
    s: &[T],
    f: &dyn Fn() -> Fields,
) -> Result<ArrowTable> 
where
    T: Sized + Serialize,
{
    let batch_size = s.iter().len();
    let bytes = serde_json::to_vec(s)?;
    ArrowTableBuilder::new()
        .with_name(name)
        .with_schema(create_schema_from_fields(f))
        .with_json(&bytes, batch_size)?
        .build()
}

pub fn extract_struct_from_table<T>(table: &ArrowTable) -> Result<Vec<T>>
where
    T: Sized + for<'a> Deserialize<'a>
{
    let bytes = table.to_json()?;
    let content = match serde_json::from_slice::<Vec<T>>(&bytes) {
        Ok(content) => content,
        Err(err) => return Err(anyhow!("{err}")),
    };
    Ok(content)
}

pub fn create_messages_fields() -> Fields {
    let field_names = ["role", "content"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec.push(Field::new("timestamp", DataType::Int64, false));
    Fields::from(fields_vec)
}

/// In combination with [MessagesTraitExt]
/// 
/// MessagesTraitExt: phymes-core/src/schemas/messages.rs
pub struct MessagesSubject {
    pub role: String,
    pub content: String,
    pub timestamp: i64,
}

pub fn create_messages_record_batch(
    role: Vec<String>,
    content: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let role: ArrayRef = Arc::new(StringArray::from(role));
    let content: ArrayRef = Arc::new(StringArray::from(content));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("role", role),
        ("content", content),
        ("timestamp", timestamp),
    ])?;
    Ok(batch)
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

pub fn create_queries_fields() -> Fields {
    let field_names = ["query_id", "text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub struct QueriesSubject {
    pub query_id: String,
    pub text: String,
}

impl QueriesSubject {
    pub fn new(text: &str) -> Self {
        let content = if cfg!(feature = "hf_hub") {
            // DM: note that the prompt for the query is specific to Qwen!
            format!(
                "{}{}",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                text
            )
        } else {
            text.to_string()
        };
        Self { query_id: create_timestamp_str(), text: content }
    }
}

pub fn create_queries_batch(
    query_ids: Vec<String>,
    text: Vec<String>,
) -> Result<RecordBatch> {
    let query_ids: ArrayRef = Arc::new(StringArray::from(query_ids));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let batch = RecordBatch::try_from_iter(vec![
        ("query_id", query_ids),
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

pub fn create_blob_fields() -> Fields {
    let filename = Field::new("filename", DataType::Utf8, false);
    let extension = Field::new("extension", DataType::Utf8, false);
    let list_data_type = DataType::List(
        Arc::new(Field::new_list_field(DataType::UInt8, false))
    );
    let bytes = Field::new("bytes", list_data_type, false);
    let metadata = Field::new("metadata", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    Fields::from(vec![
        filename,
        extension,
        bytes,
        metadata,
        timestamp,
    ])
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BlobSubject {
    pub filename: String, 
    pub bytes: Vec<u8>, 
    pub extension: String, 
    pub metadata: String,
    pub timestamp: i64,
}

pub fn create_blob_batch(
    filename: Vec<String>,
    extension: Vec<String>,
    bytes: Vec<Vec<u8>>,
    metadata: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let filename: ArrayRef = Arc::new(StringArray::from(filename));
    let extension: ArrayRef = Arc::new(StringArray::from(extension));
    let value_builder = UInt8Builder::new();
    let mut list_builder = ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::UInt8, false));
    for values in bytes.into_iter() {
        list_builder.values().append_slice(&values);
        list_builder.append(true);
    }
    let bytes: ArrayRef = Arc::new(list_builder.finish());
    let metadata: ArrayRef = Arc::new(StringArray::from(metadata));
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("filename", filename),
        ("extension", extension),
        ("bytes", bytes),
        ("metadata", metadata),
        ("timestamp", timestamp),
    ])?;
    Ok(batch)
}

pub trait AvailableSubjectsTrait {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<ArrowTable>;
    fn to_table_from_struct<T>(&self, name: Option<&str>, s: &[T]) -> Result<ArrowTable> where T: Sized + Serialize;
    fn to_struct_from_table<T>(&self, table: &ArrowTable) -> Result<Vec<T>> where T: Sized + for<'a> Deserialize<'a>;
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
        }
    }
}

impl AvailableSubjectsTrait for AvailableSubjects {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<ArrowTable> {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        match self {
            AvailableSubjects::Messages => {
                create_table_from_fields(name.as_str(), batches, &create_messages_fields)
            }
            AvailableSubjects::Values => {
                create_table_from_fields(name.as_str(), batches, &create_values_fields)
            }
            AvailableSubjects::Configs => {
                create_table_from_fields(name.as_str(), batches, &create_config_fields)
            }
            AvailableSubjects::Tools => {
                create_table_from_fields(name.as_str(), batches, &create_tools_fields)
            }
            AvailableSubjects::Documents => {
                create_table_from_fields(name.as_str(), batches, &create_documents_fields)
            }
            AvailableSubjects::Queries => {
                create_table_from_fields(name.as_str(), batches, &create_queries_fields)
            }
            AvailableSubjects::DocumentEmbeddings => create_table_from_fields(
                name.as_str(), batches,
                &create_document_embeddings_fields,
            ),
            AvailableSubjects::QueryEmbeddings => create_table_from_fields(
                name.as_str(), batches,
                &create_query_embeddings_fields,
            ),
            AvailableSubjects::EmbeddingScores => create_table_from_fields(
                name.as_str(), batches,
                &create_embeddings_scores_fields,
            ),
            AvailableSubjects::JoinChunksScores => create_table_from_fields(
                name.as_str(), batches,
                &create_join_chunks_scores_fields,
            ),
            AvailableSubjects::Blob => create_table_from_fields(
                name.as_str(), batches,
                &create_blob_fields,
            ),
        }
    }
    fn to_table_from_struct<T>(&self, name: Option<&str>, s: &[T]) -> Result<ArrowTable> where T: Sized + Serialize {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        match self {
            AvailableSubjects::Messages => create_table_from_fields_and_struct::<T>(name.as_str(), s, &create_messages_fields),
            AvailableSubjects::Values => create_table_from_fields_and_struct::<T>(name.as_str(), s, &create_values_fields),
            AvailableSubjects::Configs => create_table_from_fields_and_struct::<T>(name.as_str(), s, &create_config_fields),
            AvailableSubjects::Tools => create_table_from_fields_and_struct::<T>(name.as_str(), s, &create_tools_fields),
            AvailableSubjects::Documents => create_table_from_fields_and_struct::<T>(name.as_str(), s, &create_documents_fields),
            AvailableSubjects::Queries => create_table_from_fields_and_struct::<T>(name.as_str(), s, &create_queries_fields),
            AvailableSubjects::DocumentEmbeddings => create_table_from_fields_and_struct::<T>(
                name.as_str(), s,
                &create_document_embeddings_fields,
            ),
            AvailableSubjects::QueryEmbeddings => create_table_from_fields_and_struct::<T>(
                name.as_str(), s,
                &create_query_embeddings_fields,
            ),
            AvailableSubjects::EmbeddingScores => create_table_from_fields_and_struct::<T>(
                name.as_str(), s,
                &create_embeddings_scores_fields,
            ),
            AvailableSubjects::JoinChunksScores => create_table_from_fields_and_struct::<T>(
                name.as_str(), s,
                &create_join_chunks_scores_fields,
            ),
            AvailableSubjects::Blob => create_table_from_fields_and_struct::<T>(
                name.as_str(), s,
                &create_blob_fields,
            ),
        }
    }
    fn to_struct_from_table<T>(&self, table: &ArrowTable) -> Result<Vec<T>> where T: Sized + for<'a> Deserialize<'a> {
        extract_struct_from_table::<T>(table)
    }
}

impl AvailableSubjects {
    pub fn to_schema(&self) -> SchemaRef {
        match self {
            AvailableSubjects::Messages => create_schema_from_fields(&create_messages_fields),
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
        }
    }
}