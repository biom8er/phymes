use anyhow::{Error, Result};
use phymes_diagnostics::{HashMap, create_timestamp_micros};

use crate::{
    AvailableSubjects, BuilderTrait, IPCMessage, IPCMessageBuilder, MappableTrait,
    MessageBuilderTrait, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilder,
    Table, TableBuilder, TableBuilderTrait, TablePublication, TableTrait, create_chat_record_batch,
};

/// Create the error table
///
/// # Arguments
/// `err` - [anyhow::Error]
/// `with_display` - whether to show the full backtrace or not
///
/// # Notes
/// - use :? and not .to_string() with Anyhow::Error to see full backtrace if available
pub fn create_error_table(err: &Error, with_display: bool) -> Result<Table> {
    let error_str = if with_display {
        format! {"{err:?}"}
    } else {
        format! {"{err}"}
    };
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
    with_display: bool,
) -> Result<HashMap<String, SendableRecordBatchStreamMessage>> {
    let table = create_error_table(err, with_display)?;
    let message = SendableRecordBatchStreamMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&TablePublication::Extend {
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
    with_display: bool,
) -> Result<HashMap<String, IPCMessage>> {
    let table = create_error_table(err, with_display)?;
    let message = IPCMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&TablePublication::Extend {
            table_name: AvailableSubjects::SessionErrors.to_string(),
        })
        .with_message(table.to_ipc_stream()?)
        .make_random_name()?
        .build()?;
    let mut message_map = HashMap::<String, IPCMessage>::new();
    let _ = message_map.insert(message.get_name().to_string(), message);
    Ok(message_map)
}
