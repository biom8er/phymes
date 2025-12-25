use std::sync::Arc;

use anyhow::Result;
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray, UInt8Array, UInt32Array},
    datatypes::{DataType, Field, Fields},
};

pub(crate) fn create_session_subjects_fields() -> Fields {
    let field_names = ["subject_name", "column_name", "type_name"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_subjects_batch(
    subject_names: Vec<String>,
    cols_names: Vec<String>,
    type_names: Vec<String>,
) -> Result<RecordBatch> {
    let subject_names: ArrayRef = Arc::new(StringArray::from(subject_names));
    let cols_names: ArrayRef = Arc::new(StringArray::from(cols_names));
    let type_names: ArrayRef = Arc::new(StringArray::from(type_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("subject_name", subject_names),
        ("column_name", cols_names),
        ("type_name", type_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_tasks_fields() -> Fields {
    let field_names = ["task_name", "processor_name", "runtime_env_name"];
    let fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    Fields::from(fields_vec)
}

pub fn create_session_tasks_batch(
    task_names: Vec<String>,
    processor_names: Vec<String>,
    runtime_env_names: Vec<String>,
) -> Result<RecordBatch> {
    let task_names: ArrayRef = Arc::new(StringArray::from(task_names));
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let runtime_env_names: ArrayRef = Arc::new(StringArray::from(runtime_env_names));
    let batch = RecordBatch::try_from_iter(vec![
        ("task_name", task_names),
        ("processor_name", processor_names),
        ("runtime_env_name", runtime_env_names),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_processors_fields() -> Fields {
    let field_names = [
        "processor_name",
        "processor_type",
        "publication_subscription_name",
        "publication_subscription_table_names",
        "subscribe_type",
    ];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["is_subscription"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt8, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_session_processors_batch(
    processor_names: Vec<String>,
    processor_types: Vec<String>,
    pub_sub_name: Vec<String>,
    pub_sub_table_names: Vec<String>,
    subscribe_types: Vec<String>,
    is_sub: Vec<u8>,
) -> Result<RecordBatch> {
    let processor_names: ArrayRef = Arc::new(StringArray::from(processor_names));
    let processor_types: ArrayRef = Arc::new(StringArray::from(processor_types));
    let pub_sub_name: ArrayRef = Arc::new(StringArray::from(pub_sub_name));
    let pub_sub_table_names: ArrayRef = Arc::new(StringArray::from(pub_sub_table_names));
    let subscribe_types: ArrayRef = Arc::new(StringArray::from(subscribe_types));
    let is_sub: ArrayRef = Arc::new(UInt8Array::from(is_sub));
    let batch = RecordBatch::try_from_iter(vec![
        ("processor_name", processor_names),
        ("processor_type", processor_types),
        ("publication_subscription_name", pub_sub_name),
        ("publication_subscription_table_names", pub_sub_table_names),
        ("subscribe_type", subscribe_types),
        ("is_subscription", is_sub),
    ])?;
    Ok(batch)
}

pub(crate) fn create_session_runtime_envs_fields() -> Fields {
    let field_names = ["runtime_env_name"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["memory_limit", "time_limit"];
    fields_vec.extend(
        field_names
            .iter()
            .map(|f| Field::new(*f, DataType::UInt32, false))
            .collect::<Vec<_>>(),
    );
    Fields::from(fields_vec)
}

pub fn create_session_runtime_envs_batch(
    runtime_env_names: Vec<String>,
    memory_limits: Vec<u32>,
    time_limits: Vec<u32>,
) -> Result<RecordBatch> {
    let runtime_env_names: ArrayRef = Arc::new(StringArray::from(runtime_env_names));
    let memory_limits: ArrayRef = Arc::new(UInt32Array::from(memory_limits));
    let time_limits: ArrayRef = Arc::new(UInt32Array::from(time_limits));
    let batch = RecordBatch::try_from_iter(vec![
        ("runtime_env_name", runtime_env_names),
        ("memory_limit", memory_limits),
        ("time_limit", time_limits),
    ])?;
    Ok(batch)
}
