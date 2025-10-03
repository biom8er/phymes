use std::{fmt::Display, sync::Arc};

use arrow::{array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array}, datatypes::{DataType, Field, Fields}};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{schemas::available_subjects::create_timestamp_str, table::table_trait::TableBuilder};

/// Logging Levels (from highest to lowest priority):
/// error!: For critical errors that cause the program to fail or behave incorrectly.
/// warn!: For warnings about potential issues that are not immediately fatal.
/// info!: For general informational messages about the program's operation.
/// debug!: For detailed debugging information, typically used during development.
/// trace!: For extremely fine-grained information, useful for tracing program execution.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub enum LogLevel {
    Trace,
    #[default]
    Debug,
    Info,
    Warn,
    Error,
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "Trace"),
            LogLevel::Debug => write!(f, "Debug"),
            LogLevel::Info => write!(f, "Info"),
            LogLevel::Warn => write!(f, "Warn"),
            LogLevel::Error => write!(f, "Error"),
        }
    }
}

/// Fields for the log where `value` is a [serde] deserializable [String]
pub fn create_logs_fields() -> Fields {
    let field_names = ["level", "value", "file"];
    let mut fields_vec = field_names
        .iter()
        .map(|f| Field::new(*f, DataType::Utf8, false))
        .collect::<Vec<_>>();
    let field_names = ["line", "column"];
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
    pub level: LogLevel,
    pub value: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub timestamp: i64,
}
