use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, ListBuilder, RecordBatch, StringArray, UInt8Builder},
    datatypes::{DataType, Field, Fields},
};
use phymes_diagnostics::create_timestamp_micros;
use serde::{Deserialize, Serialize};

/// Object storage schema
///
/// # Notes
/// ## Location components (AWS)
/// - protocol: "https" of "https://<bucket-name>.s3.amazonaws.com/<key>"
/// - endpoint: "s3.amazonaws.com" of "https://<bucket-name>.s3.amazonaws.com/<key>"
/// - bucket:
/// - key: filesystem path
/// 
/// ## Location components (filesystem)
/// - path: "/bar/foo.rs.gz"
/// - stem: "/bar/" of "/bar/foo.rs.gz"
/// - filename: "foo" of "/bar/foo.rs.gz"
/// - prefix: "rs" of "/bar/foo.rs.gz"
/// - extension: "gz" of "/bar/foo.rs.gz"
/// 
/// ## Environmental variables
/// - access keys, regions, etc. are considered private and only accessible from the environment
pub fn create_object_store_fields() -> Fields {
    let field_names = [
        "location",
        "bucket",
        "metadata",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["last_modified"];
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

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ObjectStoreSubject {
    pub location: String,
    pub bucket: String,
    pub metadata: String,
    pub bytes: Vec<u8>,
    pub last_modified: i64,
}

#[allow(clippy::too_many_arguments)]
pub fn create_object_store_batch(
    location: Vec<String>,
    bucket: Vec<String>,
    metadata: Vec<String>,
    last_modified: Vec<i64>,
    bytes: Vec<Vec<u8>>,
) -> Result<RecordBatch> {
    let location: ArrayRef = Arc::new(StringArray::from(location));
    let bucket: ArrayRef = Arc::new(StringArray::from(bucket));
    let metadata: ArrayRef = Arc::new(StringArray::from(metadata));
    let last_modified: ArrayRef = Arc::new(Int64Array::from(last_modified));
    let value_builder = UInt8Builder::new();
    let mut list_builder =
        ListBuilder::new(value_builder).with_field(Field::new_list_field(DataType::UInt8, false));
    for values in bytes.into_iter() {
        list_builder.values().append_slice(&values);
        list_builder.append(true);
    }
    let bytes: ArrayRef = Arc::new(list_builder.finish());
    let batch = RecordBatch::try_from_iter(vec![
        ("location", location),
        ("bucket", bucket),
        ("metadata", metadata),
        ("last_modified", last_modified),
        ("bytes", bytes),
    ])?;
    Ok(batch)
}

/// Object store metadata fields
/// see <https://docs.rs/object_store/latest/object_store/struct.ObjectMeta.html>
pub fn create_object_store_meta_fields() -> Fields {
    let field_names = [
        "location",
        "e_tag",
        "version",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["size"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    let field_names = ["last_modified"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}