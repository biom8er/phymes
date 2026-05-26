use anyhow::Result;
use arrow::{
    array::{ArrayRef, Float32Builder, ListBuilder, StringArray},
    datatypes::{DataType, Field, Fields},
    record_batch::RecordBatch,
};
use std::sync::Arc;

pub(crate) fn create_documents_fields_vec() -> Vec<Field> {
    let field_names = ["chunk_id", "document_id", "text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec
}

pub fn create_documents_fields() -> Fields {
    Fields::from_iter(create_documents_fields_vec())
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
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)));
    let embedding = Field::new("embedding", list_data_type, false);
    Fields::from(vec![chunk_id, document_id, embedding])
}

pub fn create_documents_embeddings_batch(
    chunk_id: Vec<String>,
    document_id: Vec<String>,
    embedding: Vec<Vec<f32>>,
) -> Result<RecordBatch> {
    let chunk_id: ArrayRef = Arc::new(StringArray::from(chunk_id));
    let document_id: ArrayRef = Arc::new(StringArray::from(document_id));
    let value_builder = Float32Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Float32, false));
    for values in embedding.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let embedding: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("chunk_id", chunk_id),
        ("document_id", document_id),
        ("embedding", embedding),
    ])?;
    Ok(batch)
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
    Fields::from(vec![chunk_id, query_id, score, document_id, text])
}
