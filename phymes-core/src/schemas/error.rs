use anyhow::{Error, Result};
use phymes_diagnostics::{HashMap, create_timestamp_micros};

use crate::{
    create_chat_record_batch, schemas::available_subjects::AvailableSubjects, session::{BuilderTrait, MappableTrait}, table::{Table, TableBuilder, TableBuilderTrait, TablePublish, TableTrait}, task::{
        IPCMessage, IPCMessageBuilder, MessageBuilderTrait, SendableRecordBatchStreamMessage,
        SendableRecordBatchStreamMessageBuilder,
    }
};

pub fn create_error_table(err: &Error) -> Result<Table> {
    // DM: must use :? and not .to_string() with Anyhow::Error to see full backtrace if available
    let error_str = format! {"{err:?}"};
    let batch = create_chat_record_batch(
        vec!["tool".to_string()],
        vec![error_str],
        vec![create_timestamp_micros()],
    )?;
    TableBuilder::new()
        .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
        .with_record_batches(vec![batch])?
        .build()
}

pub fn create_error_message_map_stream(
    err: &Error,
    publisher: &str,
) -> Result<HashMap<String, SendableRecordBatchStreamMessage>> {
    let table = create_error_table(err)?;
    let message = SendableRecordBatchStreamMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&TablePublish::Extend {
            table_name: AvailableSubjects::SessionErrors.to_string(),
        })
        .with_message(table.to_record_batch_stream())
        .make_name()?
        .build()?;
    let mut message_map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message_map.insert(message.get_name().to_string(), message);
    Ok(message_map)
}

pub fn create_error_message_map(
    err: &Error,
    publisher: &str,
) -> Result<HashMap<String, IPCMessage>> {
    let table = create_error_table(err)?;
    let message = IPCMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&TablePublish::Extend {
            table_name: AvailableSubjects::SessionErrors.to_string(),
        })
        .with_message(table.to_ipc_stream()?)
        .make_random_name()?
        .build()?;
    let mut message_map = HashMap::<String, IPCMessage>::new();
    let _ = message_map.insert(message.get_name().to_string(), message);
    Ok(message_map)
}
