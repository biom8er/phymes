use anyhow::Result;
use clap::ValueEnum;
use phymes_core::{
    BuildableTrait, BuilderTrait, DataFormat, MappableTrait, SendableRecordBatchStream, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_diagnostics::{HashMap, TraceableTrait, Tracer};
use phymes_schemas::{create_bytes_record_batch, create_route_bytes_fields};
use phymes_event::Publication;

use crate::{IPCMessageBuilder, IPCMessageMap, MessageBuilderTrait, SendableRecordBatchStreamMessageBuilder};

/// An [RecordBatch], `IPCStream`, or [SendableRecordBatch] with additional
/// metadata for subject, publisher, and update
///
/// [SendableRecordBatch]: crate::SendableRecordBatchStream
/// [RecordBatch]: arrow::record_batch::RecordBatch
pub trait MessageTrait: MappableTrait + BuildableTrait + Send {
    type T;
    fn get_subject(&self) -> &str;
    fn get_publisher(&self) -> &str;
    fn get_update(&self) -> &Publication;
    fn get_message(&self) -> &<Self as MessageTrait>::T;
    fn get_message_own(self) -> <Self as MessageTrait>::T;
    fn get_message_mut(&mut self) -> &mut <Self as MessageTrait>::T;
}

#[derive(Clone, Default, Debug)]
pub struct IPCMessage {
    /// Name of the message
    pub(crate) name: String,
    /// The name of the subject
    pub(crate) subject: String,
    /// The name of the publishing task
    pub(crate) publisher: String,
    /// The actual message as an IPC stream
    pub(crate) message: Vec<u8>,
    /// How to update the state
    pub(crate) update: Publication,
}

impl IPCMessage {
    pub fn new(
        name: &str,
        subject: &str,
        publisher: &str,
        message: Option<Vec<u8>>,
        update: Option<Publication>,
    ) -> Self {
        Self {
            name: name.to_string(),
            subject: subject.to_string(),
            publisher: publisher.to_string(),
            message: message.unwrap_or_default(),
            update: update.unwrap_or_default(),
        }
    }

    /// Convert the message to a message map
    ///
    /// # Note
    /// ## Routing to multiple subjects
    ///
    /// - Each row in the message will be allocated to a new message when
    ///   the `bytes` [RecordBatch] schema is followed which includes columns
    ///   for `name`, `publisher`, `subject`, `format`,  and `bytes`, and
    ///   where `bytes` is a serializable payload
    /// - It is up to the implementer to assure that the `values`
    ///   can be deserialized to the intended schema
    ///
    /// [RecordBatch]: arrow::record_batch::RecordBatch
    pub fn to_map(self) -> Result<IPCMessageMap> {
        // Expected fields if it is an aggregated message
        let fields = create_route_bytes_fields();

        // Wrap the message in a table
        let table = SubjectBuilder::new_from_ipc_stream(&self.message)?
            .with_name(&self.subject)
            .build()?;

        if table.get_schema().fields().contains(&fields) {
            let names = table.get_column_as_vec_str("name");
            let publishers = table.get_column_as_vec_str("publisher");
            let subjects = table.get_column_as_vec_str("subject");
            let formats = table
                .get_column_as_vec_str("format")
                .into_iter()
                .map(|s| DataFormat::from_str(s, false).unwrap())
                .collect::<Vec<_>>();
            let bytes = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
            let map: Result<HashMap<String, IPCMessage>> = names
                .into_iter()
                .zip(publishers)
                .zip(subjects)
                .zip(formats)
                .zip(bytes)
                .map(|((((name, publisher), subject), format), bytes)| {
                    let batch = create_bytes_record_batch(vec![bytes])?;
                    let values = SubjectBuilder::new()
                        .with_name(name)
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_ipc_stream()?;
                    let message = IPCMessageBuilder::new()
                        .with_publisher(publisher)
                        .with_subject(subject)
                        .with_update(&Publication::ExtendBytes {
                            subject_name: subject.to_string(),
                            col_name: "bytes".to_string(),
                            serialize_format: format,
                        })
                        .with_message(values)
                        .make_name()?
                        .build()?;
                    Ok((message.get_name().to_string(), message))
                })
                .collect();
            map
        } else {
            // No need to split the message
            let mut map = HashMap::<String, IPCMessage>::new();
            let _ = map.insert(self.name.clone(), self);
            Ok(map)
        }
    }
}

impl MappableTrait for IPCMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for IPCMessage {
    type T = IPCMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl TraceableTrait for IPCMessage {
    fn to_trace(&self) -> Tracer {
        Tracer::new(&self.name, &self.subject)
    }
}

impl MessageTrait for IPCMessage {
    type T = Vec<u8>;
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &Publication {
        &self.update
    }
    fn get_message(&self) -> &<Self as MessageTrait>::T {
        &self.message
    }
    fn get_message_own(self) -> <Self as MessageTrait>::T {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut <Self as MessageTrait>::T {
        &mut self.message
    }
}

pub struct SendableRecordBatchStreamMessage {
    /// Name of the message
    pub(crate) name: String,
    /// The name of the intended subject task
    pub(crate) subject: String,
    /// The name of the publisher task
    pub(crate) publisher: String,
    /// The actual message
    pub(crate) message: SendableRecordBatchStream,
    /// How to update the state
    pub(crate) update: Publication,
}

impl MappableTrait for SendableRecordBatchStreamMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for SendableRecordBatchStreamMessage {
    type T = SendableRecordBatchStreamMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl TraceableTrait for SendableRecordBatchStreamMessage {
    fn to_trace(&self) -> Tracer {
        Tracer::new(&self.name, &self.subject)
    }
}

impl MessageTrait for SendableRecordBatchStreamMessage {
    type T = SendableRecordBatchStream;
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &Publication {
        &self.update
    }
    fn get_message(&self) -> &<Self as MessageTrait>::T {
        &self.message
    }
    fn get_message_own(self) -> <Self as MessageTrait>::T {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut <Self as MessageTrait>::T {
        &mut self.message
    }
}

/// Remove a message from a [HashMap] of [MessageTrait]s indexed by `message_name` by the message's `subject_name`
pub fn remove_message_by_subject<T>(subject: &str, messages: &mut HashMap<String, T>) -> Option<T>
where
    T: MessageTrait,
{
    let subjects = messages
        .iter()
        .filter_map(|(k, v)| {
            if subject == v.get_subject() {
                Some(k)
            } else {
                None
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    if let Some(s) = subjects.first() {
        messages.remove(s)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{SubjectBuilder, SubjectTrait, test_subject};
    use phymes_diagnostics::HashMap;
    use phymes_schemas::create_route_bytes_record_batch;

    use super::*;

    #[test]
    fn test_input_message_to_map() -> Result<()> {
        let test_table_1 = test_subject::make_test_subject("data", 4, 0, 3)?;
        let test_table_2 = test_subject::make_test_subject_chat("chat")?;
        let names = ["data", "chat"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let publishers = ["s1", "s2"].into_iter().map(|s| s.to_string()).collect();
        let subjects = ["d1", "d2"].into_iter().map(|s| s.to_string()).collect();
        let formats = [DataFormat::Ipc, DataFormat::Bytes]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let bytes = vec![
            test_table_1.to_ipc_stream()?,
            test_table_2.to_bytes()?.to_vec(),
        ];
        let batch = create_route_bytes_record_batch(names, publishers, subjects, formats, bytes)?;
        let table = SubjectBuilder::new()
            .with_name("")
            .with_record_batches(vec![batch])?
            .build()?;
        let message = IPCMessageBuilder::new()
            .with_name("")
            .with_publisher("")
            .with_subject("")
            .with_update(&Publication::None)
            .with_message(table.to_ipc_stream()?)
            .build()?;
        let message_map = message.to_map()?;
        assert_eq!(message_map.len(), 2);
        assert_eq!(
            message_map.get("from_s1_on_d1").unwrap().get_name(),
            "from_s1_on_d1"
        );
        assert_eq!(
            message_map.get("from_s1_on_d1").unwrap().get_publisher(),
            "s1"
        );
        assert_eq!(
            message_map.get("from_s1_on_d1").unwrap().get_subject(),
            "d1"
        );
        assert_eq!(
            *message_map.get("from_s1_on_d1").unwrap().get_update(),
            Publication::ExtendBytes {
                subject_name: "d1".to_string(),
                col_name: "bytes".to_string(),
                serialize_format: DataFormat::Ipc
            }
        );
        let test_table = SubjectBuilder::new_from_ipc_stream(
            message_map.get("from_s1_on_d1").unwrap().get_message(),
        )?
        .with_name("")
        .build()?;
        let test_bytes = test_table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let expected_bytes = test_table_1.to_ipc_stream()?;
        assert_eq!(test_bytes, expected_bytes);
        assert_eq!(
            message_map.get("from_s2_on_d2").unwrap().get_name(),
            "from_s2_on_d2"
        );
        assert_eq!(
            message_map.get("from_s2_on_d2").unwrap().get_publisher(),
            "s2"
        );
        assert_eq!(
            message_map.get("from_s2_on_d2").unwrap().get_subject(),
            "d2"
        );
        assert_eq!(
            *message_map.get("from_s2_on_d2").unwrap().get_update(),
            Publication::ExtendBytes {
                subject_name: "d2".to_string(),
                col_name: "bytes".to_string(),
                serialize_format: DataFormat::Bytes
            }
        );
        let test_table = SubjectBuilder::new_from_ipc_stream(
            message_map.get("from_s2_on_d2").unwrap().get_message(),
        )?
        .with_name("")
        .build()?;
        let test_bytes = test_table
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let expected_bytes = test_table_2.to_bytes()?.to_vec();
        assert_eq!(test_bytes, expected_bytes);
        Ok(())
    }

    #[test]
    fn test_remove_message_by_subject() -> Result<()> {
        // Test data
        let test_table = test_subject::make_test_subject("test_table", 4, 8, 3)?;

        // Test messages
        let mut messages = HashMap::<String, IPCMessage>::new();
        let message = IPCMessageBuilder::new()
            .with_subject("subject_1")
            .with_publisher("publisher")
            .with_update(&Publication::Extend {
                subject_name: "subject_1".to_string(),
            })
            .with_message(test_table.to_ipc_stream()?)
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let message = IPCMessageBuilder::new()
            .with_subject("subject_2")
            .with_publisher("publisher")
            .with_update(&Publication::None)
            .with_message(test_table.to_ipc_stream()?)
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let message = IPCMessageBuilder::new()
            .with_subject("subject_1")
            .with_publisher("publisher")
            .with_update(&Publication::None)
            .with_message(test_table.to_ipc_stream()?)
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);

        // Test that we can get both subjects with the same name
        let m = remove_message_by_subject("subject_1", &mut messages).unwrap();
        assert_eq!(m.get_publisher(), "publisher");
        assert_eq!(m.get_subject(), "subject_1");
        let m = remove_message_by_subject("subject_1", &mut messages).unwrap();
        assert_eq!(m.get_publisher(), "publisher");
        assert_eq!(m.get_subject(), "subject_1");
        let m = remove_message_by_subject("subject_1", &mut messages);
        assert!(m.is_none());

        Ok(())
    }
}
