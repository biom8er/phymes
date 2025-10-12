use std::{fmt::Display, sync::Arc};

use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Fields for the log where `value` is a [serde] deserializable [String]
/// `parent_name` will most likely be the task name
/// `span_name` will most likely be the processor name
/// `message_name` will most likely be the subject name
pub fn create_trace_fields() -> Fields {
    let field_names = ["parent_name", "span_name", "message_name", "direction", "file"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["line", "parent_id", "span_id"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    fields_vec.push(Field::new("timestamp", DataType::Int64, false));
    Fields::from(fields_vec)
}

/// Fields for the log where `value` is a [serde] deserializable [String]
/// `span_id` connects the trace data to the event data
pub fn create_events_fields() -> Fields {
    let field_names = ["level", "value", "span_name", "file"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["line", "column", "span_id", "id"];
    fields_vec.extend(field_names
        .iter()
        .map(|f| Field::new(*f, DataType::UInt32, false))
        .collect::<Vec<_>>());
    fields_vec.push(Field::new("timestamp", DataType::Int64, false));
    Fields::from(fields_vec)
}

pub fn create_logs_batch(
    level: Vec<String>,
    value: Vec<String>,
    file: Vec<String>,
    line: Vec<u32>,
    column: Vec<u32>,
    timestamp: Vec<i64>,
) -> Result<RecordBatch> {
    let level_arr: ArrayRef = Arc::new(StringArray::from(level));
    let value_arr: ArrayRef = Arc::new(StringArray::from(value));
    let file_arr: ArrayRef = Arc::new(StringArray::from(file));
    let line_arr: ArrayRef = Arc::new(UInt32Array::from(line));
    let column_arr: ArrayRef = Arc::new(UInt32Array::from(column));
    let timestamp_arr: ArrayRef = Arc::new(Int64Array::from(timestamp));
    let batch = RecordBatch::try_from_iter(vec![
        ("level", level_arr),
        ("value", value_arr),
        ("file", file_arr),
        ("line", line_arr),
        ("column", column_arr),
        ("timestamp", timestamp_arr),
    ])?;
    Ok(batch)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogSubject {
    pub level: EventLevel,
    pub value: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub timestamp: i64,
}