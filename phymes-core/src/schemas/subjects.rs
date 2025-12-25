use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt64Array},
    datatypes::{DataType, Field, Fields},
};

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
            .map(|f| Field::new(*f, DataType::UInt64, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_subjects_num_rows_batch(
    subject_names: Vec<String>,
    num_rows: Vec<u64>,
) -> Result<RecordBatch> {
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let num_rows: ArrayRef = Arc::new(UInt64Array::from(num_rows));
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
    let field_names = ["timestamp"];
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
    timestamps: Vec<i64>,
) -> Result<RecordBatch> {
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let session_names: ArrayRef = Arc::new(StringArray::from(session_names));
    let timestamps: ArrayRef = Arc::new(Int64Array::from(timestamps));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_names),
        ("task_name", task_names),
        ("session_name", session_names),
        ("timestamp", timestamps),
    ])?;
    Ok(batch)
}
