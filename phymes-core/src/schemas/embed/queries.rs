use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Float32Builder, ListBuilder, RecordBatch, StringArray},
    datatypes::{DataType, Field, Fields},
};
use serde::{Deserialize, Serialize};

pub fn create_queries_fields() -> Fields {
    let field_names = ["query_id", "text"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_queries_batch(query_ids: Vec<String>, text: Vec<String>) -> Result<RecordBatch> {
    let query_ids: ArrayRef = Arc::new(StringArray::from(query_ids));
    let text: ArrayRef = Arc::new(StringArray::from(text));
    let batch = RecordBatch::try_from_iter(vec![("query_id", query_ids), ("text", text)])?;
    Ok(batch)
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueriesSubject {
    pub query_id: String,
    pub text: String,
}

pub fn create_query_embeddings_fields() -> Fields {
    let query_id = Field::new("query_id", DataType::Utf8, false);
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::Float32, false)));
    let embedding = Field::new("embedding", list_data_type, false);
    Fields::from(vec![query_id, embedding])
}

pub fn create_query_embeddings_batch(
    query_id: Vec<String>,
    embedding: Vec<Vec<f32>>,
) -> Result<RecordBatch> {
    let query_id: ArrayRef = Arc::new(StringArray::from(query_id));
    let value_builder = Float32Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::Float32, false));
    for values in embedding.into_iter() {
        list_builder.values().append_slice(values.as_slice());
        list_builder.append(true);
    }
    let embedding: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("query_id", query_id),
        ("embedding", embedding),
    ])?;
    Ok(batch)
}
