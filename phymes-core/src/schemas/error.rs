use std::sync::Arc;

use arrow::{array::{ArrayRef, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub fn create_error_fields() -> Fields {
    let error = Field::new("error", DataType::Utf8, false);
    Fields::from(vec![error])
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ErrorSubject {
    pub error: String, 
    pub bytes: Vec<u8>, 
    pub extension: String, 
    pub metadata: String,
    pub timestamp: i64,
}

pub fn create_error_batch(error: Vec<String>) -> Result<RecordBatch> {
    let error: ArrayRef = Arc::new(StringArray::from(error));
    let batch = RecordBatch::try_from_iter(vec![("error", error)])?;
    Ok(batch)
}