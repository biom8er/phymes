use anyhow::{Error, Result};
use phymes_subject::{BuilderTrait, MappableTrait, SubjectTrait};
use phymes_diagnostics::HashMap;
use phymes_event::Publication;
use phymes_schemas::{AvailableSubjects, create_error_subject};

use crate::{
    IPCMessage, IPCMessageBuilder, MessageBuilderTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder,
};

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
