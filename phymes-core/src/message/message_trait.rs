use std::sync::Arc;

use crate::{IPCMessageBuilder, MessageBuilderTrait, SendableRecordBatchStreamMessageBuilder, BuildableTrait, BuilderTrait, IPCMessageMap, MappableTrait, SendableRecordBatchStream, TableBuilder, TableBuilderTrait, TablePublication, TableTrait};

use anyhow::Result;
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Fields};
use phymes_diagnostics::{HashMap, TraceableTrait, Tracer};

/// An [RecordBatch], `IPCStream`, or [SendableRecordBatch] with additional
/// metadata for subject, publisher, and update
///
/// [SendableRecordBatch]: crate::table::SendableRecordBatchStream
pub trait MessageTrait: MappableTrait + BuildableTrait + Send {
    type T;
    fn get_subject(&self) -> &str;
    fn get_publisher(&self) -> &str;
    fn get_update(&self) -> &TablePublication;
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
    pub(crate) update: TablePublication,
}

impl IPCMessage {
    pub fn new(
        name: &str,
        subject: &str,
        publisher: &str,
        message: Option<Vec<u8>>,
        update: Option<TablePublication>,
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
    /// Each row in the message will be allocated to
    ///   a new message if an aggregated message schema is
    ///   followed whereby there are columns for the
    ///   `name`, `publisher`, `subject`,  and `values`,
    ///   where `values` is a deserializable JSON payload
    ///
    /// # Note
    ///
    /// - It is up to the implementer to assure that the `values`
    ///   can be deserialized to either an ArrowTable or
    ///   a user-defined schema
    pub fn to_map(self) -> Result<IPCMessageMap> {
        let mut map = HashMap::<String, IPCMessage>::new();

        // Expected fields if it is an aggregated message
        let field_names = ["name", "publisher", "subject", "values"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let fields = Fields::from(fields_vec);

        // Wrap the message in a table
        let table = TableBuilder::new_from_ipc_stream(&self.message)?
            .with_name(&self.subject)
            .build()?;

        if table.get_schema().fields().contains(&fields) {
            // Each row is a new message
            let data = field_names
                .iter()
                .map(|f| table.get_column_as_vec_str(f))
                .collect::<Vec<_>>();
            let n_rows: usize = table
                .get_record_batches()
                .iter()
                .map(|batches| batches.num_rows())
                .sum::<usize>();
            for row in 0..n_rows {
                let name = data.first().unwrap().get(row).unwrap();
                let values: ArrayRef = Arc::new(StringArray::from(vec![
                    data.get(3).unwrap().get(row).unwrap().to_string(),
                ]));
                let batch = RecordBatch::try_from_iter(vec![("values", values)])?;
                let bytes = TableBuilder::new()
                    .with_name(name)
                    .with_record_batches(vec![batch])?
                    .build()?
                    .to_ipc_stream()?;
                let message = IPCMessageBuilder::new()
                    // .with_name(name)
                    .with_publisher(data.get(1).unwrap().get(row).unwrap())
                    .with_subject(data.get(2).unwrap().get(row).unwrap())
                    .with_update(&TablePublication::Extend {
                        table_name: data.get(2).unwrap().get(row).unwrap().to_string(),
                    })
                    .with_message(bytes)
                    .make_name()?
                    .build()?;
                let _ = map.insert(name.to_string(), message);
            }
        } else {
            // No need to split the message
            let _ = map.insert(self.name.clone(), self);
        }
        Ok(map)
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
    fn get_update(&self) -> &TablePublication {
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
    pub(crate) update: TablePublication,
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
    fn get_update(&self) -> &TablePublication {
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
    use std::sync::Arc;

    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_diagnostics::HashMap;

    use crate::{
        TableBuilder, TableTrait,
        test_table::{self, make_test_table, make_test_table_chat},
    };

    use super::*;

    #[test]
    fn test_input_message_to_map() -> Result<()> {
        let test_table_1 = make_test_table("data", 4, 0, 3)?;
        let json_str_1 = String::from_utf8(test_table_1.to_json()?)?;
        let test_table_2 = make_test_table_chat("chat")?;
        let json_str_2 = String::from_utf8(test_table_2.to_json()?)?;
        let names: ArrayRef = Arc::new(StringArray::from(vec!["data", "chat"]));
        let publishers: ArrayRef = Arc::new(StringArray::from(vec!["s1", "s2"]));
        let subjects: ArrayRef = Arc::new(StringArray::from(vec!["d1", "d2"]));
        let values: ArrayRef = Arc::new(StringArray::from(vec![
            json_str_1.clone(),
            json_str_2.clone(),
        ]));
        let batch = RecordBatch::try_from_iter(vec![
            ("name", names),
            ("publisher", publishers),
            ("subject", subjects),
            ("values", values),
        ])?;
        let table = TableBuilder::new()
            .with_name("")
            .with_record_batches(vec![batch])?
            .build()?;
        let message = IPCMessageBuilder::new()
            .with_name("")
            .with_publisher("")
            .with_subject("")
            .with_update(&TablePublication::None)
            .with_message(table.to_ipc_stream()?)
            .build()?;
        let message_map = message.to_map()?;
        assert_eq!(message_map.len(), 2);
        assert_eq!(message_map.get("data").unwrap().get_name(), "from_s1_on_d1");
        assert_eq!(message_map.get("data").unwrap().get_publisher(), "s1");
        assert_eq!(message_map.get("data").unwrap().get_subject(), "d1");
        assert_eq!(
            *message_map.get("data").unwrap().get_update(),
            TablePublication::Extend {
                table_name: "d1".to_string()
            }
        );
        let test_table =
            TableBuilder::new_from_ipc_stream(message_map.get("data").unwrap().get_message())?
                .with_name("")
                .build()?;
        assert_eq!(
            *test_table.get_column_as_vec_str("values").first().unwrap(),
            json_str_1
        );
        assert_eq!(message_map.get("chat").unwrap().get_name(), "from_s2_on_d2");
        assert_eq!(message_map.get("chat").unwrap().get_publisher(), "s2");
        assert_eq!(message_map.get("chat").unwrap().get_subject(), "d2");
        assert_eq!(
            *message_map.get("chat").unwrap().get_update(),
            TablePublication::Extend {
                table_name: "d2".to_string()
            }
        );
        let test_table =
            TableBuilder::new_from_ipc_stream(message_map.get("chat").unwrap().get_message())?
                .with_name("")
                .build()?;
        assert_eq!(
            *test_table.get_column_as_vec_str("values").first().unwrap(),
            json_str_2
        );

        Ok(())
    }

    #[test]
    fn test_remove_message_by_subject() -> Result<()> {
        // Test data
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Test messages
        let mut messages = HashMap::<String, IPCMessage>::new();
        let message = IPCMessageBuilder::new()
            .with_subject("subject_1")
            .with_publisher("publisher")
            .with_update(&TablePublication::Extend {
                table_name: "subject_1".to_string(),
            })
            .with_message(test_table.to_ipc_stream()?)
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let message = IPCMessageBuilder::new()
            .with_subject("subject_2")
            .with_publisher("publisher")
            .with_update(&TablePublication::None)
            .with_message(test_table.to_ipc_stream()?)
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let message = IPCMessageBuilder::new()
            .with_subject("subject_1")
            .with_publisher("publisher")
            .with_update(&TablePublication::None)
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
