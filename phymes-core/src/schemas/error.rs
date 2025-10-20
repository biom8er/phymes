use std::sync::Arc;

use arrow::{array::{ArrayRef, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::{Error, Result};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};

use crate::{schemas::available_subjects::AvailableSubjects, session::common_traits::{BuilderTrait, MappableTrait}, table::{table_publish::TablePublish, table_trait::{Table, TableBuilder, TableBuilderTrait, TableTrait}}, task::message::{IPCMessage, IPCMessageBuilder, MessageBuilderTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilder}};

pub fn create_error_fields() -> Fields {
    let error = Field::new("error", DataType::Utf8, false);
    Fields::from(vec![error])
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ErrorSubject {
    pub error: String,
}

pub fn create_error_batch(error: Vec<String>) -> Result<RecordBatch> {
    let error: ArrayRef = Arc::new(StringArray::from(error));
    let batch = RecordBatch::try_from_iter(vec![("error", error)])?;
    Ok(batch)
}

pub fn create_error_table(err: &Error) -> Result<Table> {
    // DM: must use :? and not .to_string() with Anyhow::Error to see full backtrace if available
    let error_str = format!{"{err:?}"}; 
    let batch = create_error_batch(vec![error_str])?;
    TableBuilder::new()
        .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
        .with_record_batches(vec![batch])?
        .build()
}

pub fn create_error_message_map_stream(err: &Error, publisher: &str) -> Result<HashMap<String, SendableRecordBatchStreamMessage>> {
    let table = create_error_table(err)?;
    let message = SendableRecordBatchStreamMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&TablePublish::Extend { table_name: AvailableSubjects::SessionErrors.to_string()})
        .with_message(table.to_record_batch_stream())
        .make_name()?
        .build()?;
    let mut message_map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message_map.insert(message.get_name().to_string(), message);
    Ok(message_map)
}

pub fn create_error_message_map(err: &Error, publisher: &str) -> Result<HashMap<String, IPCMessage>> {
    let table = create_error_table(err)?;
    let message = IPCMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&TablePublish::Extend { table_name: AvailableSubjects::SessionErrors.to_string()})
        .with_message(table.to_ipc_stream()?)
        .make_name()?
        .build()?;
    let mut message_map = HashMap::<String, IPCMessage>::new();
    let _ = message_map.insert(message.get_name().to_string(), message);
    Ok(message_map)
}