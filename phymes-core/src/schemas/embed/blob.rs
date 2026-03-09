use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, ListBuilder, RecordBatch, StringArray, UInt8Builder},
    datatypes::{DataType, Field, Fields},
};
use phymes_diagnostics::create_timestamp_micros;
use serde::{Deserialize, Serialize};

use crate::table::TableBuilder;

/// Attachments schema
pub fn create_attachments_fields() -> Fields {
    let filename = Field::new("filename", DataType::Utf8, false);
    let extension = Field::new("extension", DataType::Utf8, false);
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::UInt8, false)));
    let bytes = Field::new("bytes", list_data_type, false);
    let metadata = Field::new("metadata", DataType::Utf8, false);
    let timestamp = Field::new("timestamp", DataType::Int64, false);
    Fields::from(vec![filename, extension, bytes, metadata, timestamp])
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AttachmentsSubject {
    pub filename: String,
    pub bytes: Vec<u8>,
    pub extension: String,
    pub metadata: String,
    pub timestamp: i64,
}

pub fn create_attachments_batch(
    filename: Vec<String>,
    extension: Vec<String>,
    bytes: Vec<Vec<u8>>,
    metadata: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let filename: ArrayRef = Arc::new(StringArray::from(filename));
    let extension: ArrayRef = Arc::new(StringArray::from(extension));
    let value_builder = UInt8Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::UInt8, false));
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

pub trait AttachmentBuilderTraitExt: Sized {
    fn with_attachment(
        self,
        filename: Option<&str>,
        extension: Option<&str>,
        bytes: &[u8],
        metadata: Option<&str>,
    ) -> Result<Self>;
}

impl AttachmentBuilderTraitExt for TableBuilder {
    fn with_attachment(
        mut self,
        filename: Option<&str>,
        extension: Option<&str>,
        bytes: &[u8],
        metadata: Option<&str>,
    ) -> Result<Self> {
        // Handle the metadata
        let filename = filename.unwrap_or_default().to_string();
        let extension = extension.unwrap_or_default().to_string();
        let metadata = metadata.unwrap_or_default().to_string();

        // Add the record batch to the table
        let batch = create_attachments_batch(
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

/// Blob schema
///
/// # Todo
/// - Change to "Blob" after changing "Blob" to "Attachment"
///
/// # Notes
/// - path: "/bar/foo.rs.gz"
/// - stem: "/bar/" of "/bar/foo.rs.gz"
/// - filename: "foo" of "/bar/foo.rs.gz"
/// - prefix: "rs" of "/bar/foo.rs.gz"
/// - extension: "gz" of "/bar/foo.rs.gz"
pub fn create_blob_fields() -> Fields {
    let field_names = [
        "path",
        "stem",
        "filename",
        "prefix",
        "extension",
        "metadata",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["timestamp"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    let list_data_type = DataType::List(Arc::new(Field::new_list_field(DataType::UInt8, false)));
    let field_names = ["bytes"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, list_data_type.clone(), false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BlobSubject {
    pub path: String,
    pub stem: String,
    pub filename: String,
    pub prefix: String,
    pub extension: String,
    pub metadata: String,
    pub bytes: Vec<u8>,
    pub timestamp: i64,
}

#[allow(clippy::too_many_arguments)]
pub fn create_blob_batch(
    path: Vec<String>,
    stem: Vec<String>,
    filename: Vec<String>,
    prefix: Vec<String>,
    extension: Vec<String>,
    bytes: Vec<Vec<u8>>,
    metadata: Vec<String>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let path: ArrayRef = Arc::new(StringArray::from(path));
    let stem: ArrayRef = Arc::new(StringArray::from(stem));
    let filename: ArrayRef = Arc::new(StringArray::from(filename));
    let prefix: ArrayRef = Arc::new(StringArray::from(prefix));
    let extension: ArrayRef = Arc::new(StringArray::from(extension));
    let metadata: ArrayRef = Arc::new(StringArray::from(metadata));
    let value_builder = UInt8Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::UInt8, false));
    for values in bytes.into_iter() {
        list_builder.values().append_slice(&values);
        list_builder.append(true);
    }
    let bytes: ArrayRef = Arc::new(list_builder.finish());
    let timestamp: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("path", path),
        ("stem", stem),
        ("filename", filename),
        ("prefix", prefix),
        ("extension", extension),
        ("metadata", metadata),
        ("bytes", bytes),
        ("timestamp", timestamp),
    ])?;
    Ok(batch)
}
