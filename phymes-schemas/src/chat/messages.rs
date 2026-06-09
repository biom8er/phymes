use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    datatypes::{DataType, Field, Fields},
    record_batch::RecordBatch,
};

pub fn create_chat_fields() -> Fields {
    let field_names = ["role", "content"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    fields_vec.push(Field::new("timestamp", DataType::Int64, false));
    Fields::from(fields_vec)
}

#[allow(dead_code)]
pub struct ChatSubject {
    pub role: String,
    pub content: String,
    pub timestamp: i64,
}

pub fn create_chat_record_batch(
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
