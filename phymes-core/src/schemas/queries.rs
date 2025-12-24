use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Fields},
};
use serde::{Deserialize, Serialize};

pub(crate) fn create_queries_fields() -> Fields {
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
