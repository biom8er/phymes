use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array, ListBuilder, RecordBatch, StringArray, UInt8Builder}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{schemas::available_subjects::create_timestamp_micros, table::table::TableBuilder};

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

pub trait BlobBuilderTraitExt: Sized {
    fn with_blob(self, filename: Option<&str>, extension: Option<&str>, bytes: &[u8], metadata: Option<&str>) -> Result<Self>;
}

impl BlobBuilderTraitExt for TableBuilder {
    fn with_blob(mut self, filename: Option<&str>, extension: Option<&str>, bytes: &[u8], metadata: Option<&str>) -> Result<Self> {
        // Handle the metadata
        let filename = filename.unwrap_or_default().to_string();
        let extension = extension.unwrap_or_default().to_string();
        let metadata = metadata.unwrap_or_default().to_string();

        // Add the record batch to the table
        let batch = create_blob_batch(
            vec![filename],
            vec![extension],
            vec![bytes.to_vec()],
            vec![metadata],
            vec![create_timestamp_micros()],
        )?;
        match self.record_batches {
            Some(ref mut batches) => {
                batches.push(batch);
                Ok(self)
            }
            None => {
                self.schema = Some(batch.schema());
                self.record_batches = Some(vec![batch]);
                Ok(self)
            }
        }        
    }
}