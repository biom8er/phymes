use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array},
    datatypes::{DataType, Field, Fields},
};

use crate::storage::create_object_store_meta_fields_vec;

pub(crate) fn create_subjects_num_rows_fields() -> Fields {
    let field_names = ["subject_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["num_rows"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_subjects_num_rows_batch(
    subject_names: Vec<String>,
    num_rows: Vec<i64>,
) -> Result<RecordBatch> {
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let num_rows: ArrayRef = Arc::new(Int64Array::from(num_rows));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_names),
        ("num_rows", num_rows),
    ])?;
    Ok(batch)
}

pub(crate) fn create_subjects_change_log_fields() -> Fields {
    let field_names = ["subject_name", "task_name", "session_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["num_rows", "superstep"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_subjects_change_log_batch(
    subject_names: Vec<String>,
    task_names: Vec<String>,
    session_names: Vec<String>,
    num_rows: Vec<i64>,
    supersteps: Vec<i64>,
) -> Result<RecordBatch> {
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let num_rows: ArrayRef = Arc::new(Int64Array::from(num_rows));
    let supersteps: ArrayRef = Arc::new(Int64Array::from(supersteps));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_names),
        ("task_name", task_names),
        ("session_name", session_names),
        ("num_rows", num_rows),
        ("superstep", supersteps),
    ])?;
    Ok(batch)
}

pub(crate) fn create_subjects_object_store_meta_fields() -> Fields {
    let field_names = ["subject_name", "task_name", "session_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["num_rows", "superstep"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Int64, false))
            .collect::<Vec<_>>(),
    );
    fields_vec.extend(create_object_store_meta_fields_vec());
    Fields::from(fields_vec)
}

#[allow(clippy::too_many_arguments)]
pub fn create_subjects_object_store_meta_batch(
    subject_names: Vec<String>,
    task_names: Vec<String>,
    session_names: Vec<String>,
    num_rows: Vec<i64>,
    supersteps: Vec<i64>,
    location: Vec<String>,
    bucket: Vec<String>,
    e_tag: Vec<String>,
    version: Vec<String>,
    size: Vec<u32>,
    last_modified: Vec<i64>,
) -> Result<RecordBatch> {
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let num_rows: ArrayRef = Arc::new(Int64Array::from(num_rows));
    let supersteps: ArrayRef = Arc::new(Int64Array::from(supersteps));
    let location: ArrayRef = Arc::new(StringArray::from(location));
    let bucket: ArrayRef = Arc::new(StringArray::from(bucket));
    let e_tag: ArrayRef = Arc::new(StringArray::from(e_tag));
    let version: ArrayRef = Arc::new(StringArray::from(version));
    let size: ArrayRef = Arc::new(UInt32Array::from(size));
    let last_modified: ArrayRef = Arc::new(Int64Array::from(last_modified));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_names),
        ("task_name", task_names),
        ("session_name", session_names),
        ("num_rows", num_rows),
        ("superstep", supersteps),
        ("location", location),
        ("bucket", bucket),
        ("e_tag", e_tag),
        ("version", version),
        ("size", size),
        ("last_modified", last_modified),
    ])?;
    Ok(batch)
}

// pub(crate) fn create_group_by_subject_change_log_delta_fields() -> Fields {
//     let field_names = ["subject_name", "task_name", "session_name"];
//     let mut fields_vec = field_names
//         .iter()
//         .map(|f| Field::new(*f, DataType::Utf8, false))
//         .collect::<Vec<_>>();
//     let field_names = ["num_rows-Sum", "timestamp-Last"];
//     fields_vec.extend(
//         field_names
//             .iter()
//             .map(|f| Field::new(*f, DataType::Int64, false))
//             .collect::<Vec<_>>(),
//     );
//     Fields::from(fields_vec)
// }
