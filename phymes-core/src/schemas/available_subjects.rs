use crate::{session::common_traits::{BuilderTrait, MappableTrait}, table::arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait}};

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, ListBuilder, StringArray, UInt8Array, UInt8Builder},
    datatypes::{DataType, Field, Fields, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use clap::ValueEnum;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    f: &dyn Fn() -> Fields,
) -> Result<ArrowTable> {
    ArrowTableBuilder::new()
        .with_name(name)
        .with_schema(create_schema_from_fields(f))
        .with_record_batches(Vec::new())?
        .build()
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
    Fields::from(vec![
        filename,
        extension,
        bytes,
        metadata,
    ])
}

pub fn create_blob_batch(
    filename: Vec<String>,
    extension: Vec<String>,
    bytes: Vec<Vec<u8>>,
    metadata: Vec<String>,
) -> Result<RecordBatch> {
    let filename: ArrayRef = Arc::new(StringArray::from(filename));
    let extension: ArrayRef = Arc::new(StringArray::from(extension));
    let value_builder = UInt8Builder::new();
    let mut list_builder = ListBuilder::new(value_builder);
    for values in bytes.into_iter() {
        list_builder.values().append_slice(&values);
        list_builder.append(true);
    }
    let bytes: ArrayRef = Arc::new(list_builder.finish());
    let metadata: ArrayRef = Arc::new(StringArray::from(metadata));
    let batch = RecordBatch::try_from_iter(vec![
        ("filename", filename),
        ("extension", extension),
        ("bytes", bytes),
        ("metadata", metadata),
    ])?;
    Ok(batch)
}

/// The available subject schmeas
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableSubjects {
    Messages,
    #[default]
    Values,
    Tools,
    Documents,
    Queries,
    DocumentEmbeddings,
    QueryEmbeddings,
    EmbeddingScores,
    JoinChunksScores,
    Blob,
}

impl MappableTrait for AvailableSubjects {
    fn get_name(&self) -> &str {
        match self {
            AvailableSubjects::Messages => "Messages",
            AvailableSubjects::Values => "Values",
            AvailableSubjects::Tools => "Tools",
            AvailableSubjects::Documents => "Documents",
            AvailableSubjects::Queries => "Queries",
            AvailableSubjects::DocumentEmbeddings => "DocumentEmbeddings",
            AvailableSubjects::QueryEmbeddings => "QueryEmbeddings",
            AvailableSubjects::EmbeddingScores => "EmbeddingScores",
            AvailableSubjects::JoinChunksScores => "JoinChunksScores",
            AvailableSubjects::Blob => "Blob",
        }
    }
}

impl AvailableSubjects {
    pub fn to_table(&self, name: &str) -> Result<ArrowTable> {
        match self {
            AvailableSubjects::Messages => {
                create_table_from_fields(name, &create_messages_fields)
            }
            AvailableSubjects::Values => {
                create_table_from_fields(name, &create_values_fields)
            }
            AvailableSubjects::Tools => {
                create_table_from_fields(name, &create_tools_fields)
            }
            AvailableSubjects::Documents => {
                create_table_from_fields(name, &create_documents_fields)
            }
            AvailableSubjects::Queries => {
                create_table_from_fields(name, &create_queries_fields)
            }
            AvailableSubjects::DocumentEmbeddings => create_table_from_fields(
                name,
                &create_document_embeddings_fields,
            ),
            AvailableSubjects::QueryEmbeddings => create_table_from_fields(
                name,
                &create_query_embeddings_fields,
            ),
            AvailableSubjects::EmbeddingScores => create_table_from_fields(
                name,
                &create_embeddings_scores_fields,
            ),
            AvailableSubjects::JoinChunksScores => create_table_from_fields(
                name,
                &create_join_chunks_scores_fields,
            ),
            AvailableSubjects::Blob => create_table_from_fields(
                name,
                &create_blob_fields,
            ),
        }
    }
    pub fn to_schema(&self) -> SchemaRef {
        match self {
            AvailableSubjects::Messages => create_schema_from_fields(&create_messages_fields),
            AvailableSubjects::Values => create_schema_from_fields(&create_values_fields),
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