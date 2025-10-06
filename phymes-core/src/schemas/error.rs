use std::sync::Arc;

use arrow::{array::{ArrayRef, RecordBatch, StringArray}, datatypes::{DataType, Field, Fields}};
use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::{metrics::HashMap, session::{common_traits::{BuilderTrait, MappableTrait}, session_context::SessionContextTableNames}, table::{table_publish::TablePublish, table_trait::{TableBuilder, TableBuilderTrait, TableTrait}}, task::message::{IPCMessage, IPCMessageBuilder, MessageBuilderTrait}};

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

pub fn create_error_message_map(err: &Error) -> Result<HashMap<String, IPCMessage>> {    
    // DM: must use :? and not .to_string() with Anyhow::Error to see full backtrace if available
    let error_str = format!{"{err:?}"}; 
    let batch = create_error_batch(vec![error_str])?;
    let table = TableBuilder::new()
        .with_name(SessionContextTableNames::Errors.get_name())
        .with_record_batches(vec![batch])?
        .build()?;
    let message = IPCMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher("join_message_stream")
        .with_update(&TablePublish::Extend { table_name: SessionContextTableNames::Errors.to_string()})
        .with_message(table.to_ipc_stream()?)
        .make_name()?
        .build()?;
    let mut message_map = HashMap::<String, IPCMessage>::new();
    let _ = message_map.insert(table.get_name().to_string(), message);
    Ok(message_map)
}