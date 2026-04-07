use anyhow::{Error, Result};
use phymes_core::{BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
use phymes_message::{IPCMessage, IPCMessageBuilder, Publication, MessageBuilderTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder};
use phymes_diagnostics::{HashMap, create_timestamp_micros};
use phymes_schemas::{AvailableSubjects, create_chat_record_batch};

pub fn create_error_message_map_stream(
    err: &Error,
    publisher: &str,
    with_display: bool,
) -> Result<HashMap<String, SendableRecordBatchStreamMessage>> {
    let table = create_error_subject(err, with_display)?;
    let message = SendableRecordBatchStreamMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&Publication::Extend {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
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
    let table = create_error_subject(err, with_display)?;
    let message = IPCMessageBuilder::new()
        .with_subject(table.get_name())
        .with_publisher(publisher)
        .with_update(&Publication::Extend {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        })
        .with_message(table.to_ipc_stream()?)
        .make_random_name()?
        .build()?;
    let mut message_map = HashMap::<String, IPCMessage>::new();
    let _ = message_map.insert(message.get_name().to_string(), message);
    Ok(message_map)
}
