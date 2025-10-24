use std::sync::Arc;

use crate::session::{BuildableTrait, BuilderTrait, IPCMessageMap, MappableTrait};
use crate::table::{
    SendableRecordBatchStream, TableBuilder, TableBuilderTrait, TablePublish, TableTrait,
};

use anyhow::{Result, anyhow};
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
    fn get_update(&self) -> &TablePublish;
    fn get_message(&self) -> &<Self as MessageTrait>::T;
    fn get_message_own(self) -> <Self as MessageTrait>::T;
    fn get_message_mut(&mut self) -> &mut <Self as MessageTrait>::T;
}

#[derive(Clone, Default, Debug)]
pub struct IPCMessage {
    /// Name of the message
    name: String,
    /// The name of the subject
    subject: String,
    /// The name of the publishing task
    publisher: String,
    /// The actual message as an IPC stream
    message: Vec<u8>,
    /// How to update the state
    update: TablePublish,
}

impl IPCMessage {
    pub fn new(
        name: &str,
        subject: &str,
        publisher: &str,
        message: Option<Vec<u8>>,
        update: Option<TablePublish>,
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
                    .with_update(&TablePublish::Extend {
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
    fn get_update(&self) -> &TablePublish {
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
    name: String,
    /// The name of the intended subject task
    subject: String,
    /// The name of the publisher task
    publisher: String,
    /// The actual message
    message: SendableRecordBatchStream,
    /// How to update the state
    update: TablePublish,
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
    fn get_update(&self) -> &TablePublish {
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

pub trait MessageBuilderTrait: BuilderTrait + Send {
    type T;
    fn with_subject(self, name: &str) -> Self;
    fn with_publisher(self, name: &str) -> Self;
    fn make_name(self) -> Result<Self>
    where
        Self: Sized;
    fn make_random_name(self) -> Result<Self>
    where
        Self: Sized;
    fn with_update(self, update: &TablePublish) -> Self;
    fn with_message(self, message: <Self as MessageBuilderTrait>::T) -> Self;
    fn check_subject(&self) -> Result<()>;
}

#[derive(Default, Clone)]
pub struct IPCMessageBuilder {
    /// Name of the message
    pub name: Option<String>,
    /// The name of the intended subject task
    pub subject: Option<String>,
    /// The name of the publisher task
    pub publisher: Option<String>,
    /// The actually message
    pub message: Option<Vec<u8>>,
    /// How to update the state
    pub update: Option<TablePublish>,
}

impl BuilderTrait for IPCMessageBuilder {
    type T = IPCMessage;
    fn new() -> Self {
        Self {
            name: None,
            subject: None,
            publisher: None,
            message: None,
            update: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self
    where
        Self: Sized,
    {
        self.name = Some(name.to_string());
        self
    }
    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        self.check_subject()?;
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            subject: self.subject.unwrap_or_default(),
            publisher: self.publisher.unwrap_or_default(),
            message: self.message.unwrap(),
            update: self.update.unwrap(),
        })
    }
}

impl MessageBuilderTrait for IPCMessageBuilder {
    type T = Vec<u8>;
    fn with_subject(mut self, name: &str) -> Self {
        self.subject = Some(name.to_string());
        self
    }
    fn with_publisher(mut self, name: &str) -> Self {
        self.publisher = Some(name.to_string());
        self
    }
    fn with_update(mut self, update: &TablePublish) -> Self {
        self.update = Some(update.to_owned());
        self
    }
    fn make_name(self) -> Result<Self> {
        let publisher = match self.publisher {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without publisher name")),
        };
        let subject = match self.subject {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without subject name")),
        };
        let name = format!("from_{publisher}_on_{subject}");
        Ok(self.with_name(&name))
    }
    fn make_random_name(self) -> Result<Self>
    where
        Self: Sized,
    {
        let mut buf = [0u8; 16];
        getrandom::fill(&mut buf)?;
        let hash = u128::from_ne_bytes(buf);
        let subject = match self.subject {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without subject name")),
        };
        let name = format!("{subject}_{hash}");
        Ok(self.with_name(&name))
    }
    fn with_message(mut self, message: <Self as MessageBuilderTrait>::T) -> Self {
        self.message = Some(message);
        self
    }
    fn check_subject(&self) -> Result<()> {
        if self.update.as_ref().unwrap() != &TablePublish::None && self.subject.as_ref().unwrap() != self.update.as_ref().unwrap().get_table_name() {
            Err(anyhow!("Mismatch between provided subject {} and table publish table name {}.",
                self.subject.as_ref().unwrap(), self.update.as_ref().unwrap().get_table_name()))
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
pub struct SendableRecordBatchStreamMessageBuilder {
    /// Name of the message
    pub name: Option<String>,
    /// The name of the intended subject task
    pub subject: Option<String>,
    /// The name of the publisher task
    pub publisher: Option<String>,
    /// The actually message
    pub message: Option<SendableRecordBatchStream>,
    /// How to update the state
    pub update: Option<TablePublish>,
}

impl BuilderTrait for SendableRecordBatchStreamMessageBuilder {
    type T = SendableRecordBatchStreamMessage;
    fn new() -> Self {
        Self {
            name: None,
            subject: None,
            publisher: None,
            message: None,
            update: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self
    where
        Self: Sized,
    {
        self.name = Some(name.to_string());
        self
    }
    fn build(self) -> Result<Self::T>
    where
        Self: Sized,
    {
        self.check_subject()?;
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            subject: self.subject.unwrap_or_default(),
            publisher: self.publisher.unwrap_or_default(),
            message: self.message.unwrap(),
            update: self.update.unwrap(),
        })
    }
}

impl MessageBuilderTrait for SendableRecordBatchStreamMessageBuilder {
    type T = SendableRecordBatchStream;
    fn with_subject(mut self, name: &str) -> Self {
        self.subject = Some(name.to_string());
        self
    }
    fn with_publisher(mut self, name: &str) -> Self {
        self.publisher = Some(name.to_string());
        self
    }
    fn with_update(mut self, update: &TablePublish) -> Self {
        self.update = Some(update.to_owned());
        self
    }
    fn make_name(self) -> Result<Self> {
        let publisher = match self.publisher {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without publisher name")),
        };
        let subject = match self.subject {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without subject name")),
        };
        let name = format!("from_{publisher}_on_{subject}");
        Ok(self.with_name(&name))
    }
    fn make_random_name(self) -> Result<Self>
    where
        Self: Sized,
    {
        let mut buf = [0u8; 16];
        getrandom::fill(&mut buf)?;
        let hash = u128::from_ne_bytes(buf);
        let subject = match self.subject {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without subject name")),
        };
        let name = format!("{subject}_{hash}");
        Ok(self.with_name(&name))
    }
    fn with_message(mut self, message: <Self as MessageBuilderTrait>::T) -> Self {
        self.message = Some(message);
        self
    }
    fn check_subject(&self) -> Result<()> {
        if self.update.as_ref().unwrap() != &TablePublish::None && self.subject.as_ref().unwrap() != self.update.as_ref().unwrap().get_table_name() {
            Err(anyhow!("Mismatch between provided subject {} and table publish table name {}.",
                self.subject.as_ref().unwrap(), self.update.as_ref().unwrap().get_table_name()))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::table::test_table::{self, make_test_table, make_test_table_chat};

    use super::*;

    #[test]
    fn test_arrow_message_buiilders_success() -> Result<()> {
        // Test data
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Case 1: with name
        let incoming_message = IPCMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&TablePublish::Extend { table_name: "subject".to_string() })
            .with_message(test_table.to_ipc_stream()?)
            .build()?;
        assert_eq!(incoming_message.get_name(), "name");
        assert_eq!(incoming_message.get_subject(), "subject");
        assert_eq!(incoming_message.get_publisher(), "publisher");
        assert_eq!(*incoming_message.get_update(), TablePublish::Extend { table_name: "subject".to_string() });

        let outgoing_message = SendableRecordBatchStreamMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&TablePublish::Extend { table_name: "subject".to_string() })
            .with_message(test_table.to_record_batch_stream())
            .build()?;
        assert_eq!(outgoing_message.get_name(), "name");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(*outgoing_message.get_update(), TablePublish::Extend { table_name: "subject".to_string() });
        assert_eq!(
            outgoing_message.get_message().schema(),
            test_table.get_schema()
        );

        // Case 2: make name
        let incoming_message = IPCMessageBuilder::new()
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&TablePublish::None)
            .make_name()?
            .with_message(test_table.to_ipc_stream()?)
            .build()?;
        assert_eq!(incoming_message.get_name(), "from_publisher_on_subject");
        assert_eq!(incoming_message.get_subject(), "subject");
        assert_eq!(incoming_message.get_publisher(), "publisher");
        assert_eq!(*incoming_message.get_update(), TablePublish::None);

        let outgoing_message = SendableRecordBatchStreamMessageBuilder::new()
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&TablePublish::None)
            .make_name()?
            .with_message(test_table.to_record_batch_stream())
            .build()?;
        assert_eq!(outgoing_message.get_name(), "from_publisher_on_subject");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(*outgoing_message.get_update(), TablePublish::None);
        assert_eq!(
            outgoing_message.get_message().schema(),
            test_table.get_schema()
        );

        Ok(())
    }

    #[test]
    fn test_arrow_message_buiilders_mismatched_subjects() -> Result<()> {
        // Test data
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Case 1: with name
        let result = IPCMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&TablePublish::Extend { table_name: "mismatch".to_string() })
            .with_message(test_table.to_ipc_stream()?)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided subject subject and table publish table name mismatch."
            ),
        }

        let result = SendableRecordBatchStreamMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&TablePublish::Extend { table_name: "mismatch".to_string() })
            .with_message(test_table.to_record_batch_stream())
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided subject subject and table publish table name mismatch."
            ),
        }

        Ok(())
    }

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
            .with_update(&TablePublish::None)
            .with_message(table.to_ipc_stream()?)
            .build()?;
        let message_map = message.to_map()?;
        assert_eq!(message_map.len(), 2);
        assert_eq!(message_map.get("data").unwrap().get_name(), "from_s1_on_d1");
        assert_eq!(message_map.get("data").unwrap().get_publisher(), "s1");
        assert_eq!(message_map.get("data").unwrap().get_subject(), "d1");
        assert_eq!(
            *message_map.get("data").unwrap().get_update(),
            TablePublish::Extend {
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
            TablePublish::Extend {
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
}
