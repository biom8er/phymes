use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use crate::metrics::HashMap;
use crate::session::common_traits::{
    BuildableTrait, BuilderTrait, IncomingMessageMap, MappableTrait,
};
use crate::table::{
    arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
    arrow_table_publish::ArrowTablePublish,
    stream::{
        IPCRecordBatchStream, RecordBatchStream, SendableIPCRecordBatchStream,
        SendableRecordBatchStream,
    },
};

use anyhow::{Result, anyhow};
use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

/// An [RecordBatch], `IPCStream`, or [SendableRecordBatch] with additional
/// metadata for subject, publisher, and update
pub trait ArrowMessageTrait: MappableTrait + BuildableTrait + Send {
    type T;
    fn get_subject(&self) -> &str;
    fn get_publisher(&self) -> &str;
    fn get_update(&self) -> &ArrowTablePublish;
    fn get_message(&self) -> &<Self as ArrowMessageTrait>::T;
    fn get_message_own(self) -> <Self as ArrowMessageTrait>::T;
    fn get_message_mut(&mut self) -> &mut <Self as ArrowMessageTrait>::T; 
}

pub trait ArrowIncomingMessageTrait: ArrowMessageTrait {
    fn get_message(&self) -> &ArrowTable;
    fn get_message_own(self) -> ArrowTable;
    fn get_message_mut(&mut self) -> &mut ArrowTable;
}

pub trait ArrowOutgoingMessageTrait: ArrowMessageTrait {
    fn get_message(&self) -> &SendableRecordBatchStream;
    fn get_message_own(self) -> SendableRecordBatchStream;
    fn get_message_mut(&mut self) -> &mut SendableRecordBatchStream;
}

pub trait ArrowIncomingIPCMessageTrait: ArrowMessageTrait + Sync {
    fn get_message(&self) -> &Vec<u8>;
    fn get_message_own(self) -> Vec<u8>;
    fn get_message_mut(&mut self) -> &mut Vec<u8>;
}

pub trait ArrowOutgoingIPCMessageTrait: ArrowMessageTrait {
    fn get_message(&self) -> &SendableIPCRecordBatchStream;
    fn get_message_own(self) -> SendableIPCRecordBatchStream;
    fn get_message_mut(&mut self) -> &mut SendableIPCRecordBatchStream;
}

#[derive(Clone, Default, Debug)]
pub struct ArrowIncomingMessage {
    /// Name of the message
    name: String,
    /// The name of the subject
    subject: String,
    /// The name of the publishing task
    publisher: String,
    /// The actual message as an IPC stream
    message: Vec<u8>,
    /// How to update the state
    update: ArrowTablePublish,
}

impl ArrowIncomingMessage {
    pub fn new(
        name: &str,
        subject: &str,
        publisher: &str,
        message: Option<Vec<u8>>,
        update: Option<ArrowTablePublish>,
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
    pub fn to_map(self) -> Result<IncomingMessageMap> {
        let mut map = HashMap::<String, ArrowIncomingMessage>::new();

        // Expected fields if it is an aggregated message
        let field_names = ["name", "publisher", "subject", "values"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let fields = Fields::from(fields_vec);

        // Wrap the message in a table
        let table = ArrowTableBuilder::new_from_ipc_stream(&self.message)?
            .with_name(&self.subject)
            .build()?;

        if table.get_schema().fields().contains(&fields) {
            // Each row is a new message
            let data = field_names
                .iter()
                .map(|f| table.get_column_as_vec_str(f))
                .collect::<Vec<_>>();
            let n_rows: usize = table.get_record_batches()
                .iter()
                .map(|batches| batches.num_rows())
                .sum::<usize>();
            for row in 0..n_rows {
                let name = data.first().unwrap().get(row).unwrap();
                let values: ArrayRef = Arc::new(StringArray::from(vec![
                    data.get(3).unwrap().get(row).unwrap().to_string(),
                ]));
                let batch = RecordBatch::try_from_iter(vec![("values", values)])?;
                let bytes = ArrowTableBuilder::new()
                    .with_name(name)
                    .with_record_batches(vec![batch])?
                    .build()?
                    .to_ipc_stream()?;
                let message = ArrowIncomingMessageBuilder::new()
                    // .with_name(name)
                    .with_publisher(data.get(1).unwrap().get(row).unwrap())
                    .with_subject(data.get(2).unwrap().get(row).unwrap())
                    .with_update(&ArrowTablePublish::Extend {
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

impl MappableTrait for ArrowIncomingMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for ArrowIncomingMessage {
    type T = ArrowIncomingMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl ArrowMessageTrait for ArrowIncomingMessage {
    type T = Vec<u8>;
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &ArrowTablePublish {
        &self.update
    }
    fn get_message(&self) -> &<Self as ArrowMessageTrait>::T {
        &self.message
    }
    fn get_message_own(self) -> <Self as ArrowMessageTrait>::T {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut <Self as ArrowMessageTrait>::T {
        &mut self.message
    }
}

pub struct ArrowOutgoingMessage {
    /// Name of the message
    name: String,
    /// The name of the intended subject task
    subject: String,
    /// The name of the publisher task
    publisher: String,
    /// The actual message
    message: SendableRecordBatchStream,
    /// How to update the state
    update: ArrowTablePublish,
}

impl MappableTrait for ArrowOutgoingMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for ArrowOutgoingMessage {
    type T = ArrowOutgoingMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl ArrowMessageTrait for ArrowOutgoingMessage {
    type T = SendableRecordBatchStream;
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &ArrowTablePublish {
        &self.update
    }
    fn get_message(&self) -> &<Self as ArrowMessageTrait>::T {
        &self.message
    }
    fn get_message_own(self) -> <Self as ArrowMessageTrait>::T {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut <Self as ArrowMessageTrait>::T {
        &mut self.message
    }
}

pub trait ArrowMessageBuilderTrait: BuilderTrait + Send {
    type T;
    fn with_subject(self, name: &str) -> Self;
    fn with_publisher(self, name: &str) -> Self;
    fn make_name(self) -> Result<Self>
    where
        Self: Sized;
    fn make_random_name(self) -> Result<Self>
    where
        Self: Sized;
    fn with_update(self, update: &ArrowTablePublish) -> Self;
    fn with_message(self, message: <Self as ArrowMessageBuilderTrait>::T) -> Self;
}

#[derive(Default, Clone)]
pub struct ArrowIncomingMessageBuilder {
    /// Name of the message
    pub name: Option<String>,
    /// The name of the intended subject task
    pub subject: Option<String>,
    /// The name of the publisher task
    pub publisher: Option<String>,
    /// The actually message
    pub message: Option<Vec<u8>>,
    /// How to update the state
    pub update: Option<ArrowTablePublish>,
}

impl BuilderTrait for ArrowIncomingMessageBuilder {
    type T = ArrowIncomingMessage;
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
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            subject: self.subject.unwrap_or_default(),
            publisher: self.publisher.unwrap_or_default(),
            message: self.message.unwrap(),
            update: self.update.unwrap(),
        })
    }
}

impl ArrowMessageBuilderTrait for ArrowIncomingMessageBuilder {
    type T = Vec<u8>;
    fn with_subject(mut self, name: &str) -> Self {
        self.subject = Some(name.to_string());
        self
    }
    fn with_publisher(mut self, name: &str) -> Self {
        self.publisher = Some(name.to_string());
        self
    }
    fn with_update(mut self, update: &ArrowTablePublish) -> Self {
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
    fn with_message(mut self, message: <Self as ArrowMessageBuilderTrait>::T) -> Self {
        self.message = Some(message);
        self
    }
}

#[derive(Default)]
pub struct ArrowOutgoingMessageBuilder {
    /// Name of the message
    pub name: Option<String>,
    /// The name of the intended subject task
    pub subject: Option<String>,
    /// The name of the publisher task
    pub publisher: Option<String>,
    /// The actually message
    pub message: Option<SendableRecordBatchStream>,
    /// How to update the state
    pub update: Option<ArrowTablePublish>,
}

impl BuilderTrait for ArrowOutgoingMessageBuilder {
    type T = ArrowOutgoingMessage;
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
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            subject: self.subject.unwrap_or_default(),
            publisher: self.publisher.unwrap_or_default(),
            message: self.message.unwrap(),
            update: self.update.unwrap(),
        })
    }
}

impl ArrowMessageBuilderTrait for ArrowOutgoingMessageBuilder {
    type T = SendableRecordBatchStream;
    fn with_subject(mut self, name: &str) -> Self {
        self.subject = Some(name.to_string());
        self
    }
    fn with_publisher(mut self, name: &str) -> Self {
        self.publisher = Some(name.to_string());
        self
    }
    fn with_update(mut self, update: &ArrowTablePublish) -> Self {
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
    fn with_message(mut self, message: <Self as ArrowMessageBuilderTrait>::T) -> Self {
        self.message = Some(message);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::table::arrow_table::test_table::{self, make_test_table, make_test_table_chat};

    use super::*;

    #[test]
    fn test_arrow_message_buiilders() -> Result<()> {
        // Test data
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Case 1: with name
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&ArrowTablePublish::None)
            .with_message(test_table.clone())
            .build()?;
        assert_eq!(incoming_message.get_name(), "name");
        assert_eq!(incoming_message.get_subject(), "subject");
        assert_eq!(incoming_message.get_publisher(), "publisher");
        assert_eq!(*incoming_message.get_update(), ArrowTablePublish::None);
        assert_eq!(
            incoming_message.get_message().get_name(),
            test_table.get_name()
        );
        assert_eq!(
            incoming_message.get_message().get_schema(),
            test_table.get_schema()
        );

        let outgoing_message = ArrowOutgoingMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&ArrowTablePublish::None)
            .with_message(test_table.clone().to_record_batch_stream())
            .build()?;
        assert_eq!(outgoing_message.get_name(), "name");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(*outgoing_message.get_update(), ArrowTablePublish::None);
        assert_eq!(
            outgoing_message.get_message().schema(),
            test_table.get_schema()
        );

        // Case 2: make name
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&ArrowTablePublish::None)
            .make_name()?
            .with_message(test_table.clone())
            .build()?;
        assert_eq!(incoming_message.get_name(), "from_publisher_on_subject");
        assert_eq!(incoming_message.get_subject(), "subject");
        assert_eq!(incoming_message.get_publisher(), "publisher");
        assert_eq!(*incoming_message.get_update(), ArrowTablePublish::None);
        assert_eq!(
            incoming_message.get_message().get_name(),
            test_table.get_name()
        );
        assert_eq!(
            incoming_message.get_message().get_schema(),
            test_table.get_schema()
        );

        let outgoing_message = ArrowOutgoingMessageBuilder::new()
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&ArrowTablePublish::None)
            .make_name()?
            .with_message(test_table.clone().to_record_batch_stream())
            .build()?;
        assert_eq!(outgoing_message.get_name(), "from_publisher_on_subject");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(*outgoing_message.get_update(), ArrowTablePublish::None);
        assert_eq!(
            outgoing_message.get_message().schema(),
            test_table.get_schema()
        );

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
        let table = ArrowTableBuilder::new()
            .with_name("")
            .with_record_batches(vec![batch])?
            .build()?;
        let message = ArrowIncomingMessageBuilder::new()
            .with_name("")
            .with_publisher("")
            .with_subject("")
            .with_update(&ArrowTablePublish::None)
            .with_message(table)
            .build()?;
        let message_map = message.to_map()?;
        assert_eq!(message_map.len(), 2);
        assert_eq!(message_map.get("data").unwrap().get_name(), "from_s1_on_d1");
        assert_eq!(message_map.get("data").unwrap().get_publisher(), "s1");
        assert_eq!(message_map.get("data").unwrap().get_subject(), "d1");
        assert_eq!(
            *message_map.get("data").unwrap().get_update(),
            ArrowTablePublish::Extend {
                table_name: "d1".to_string()
            }
        );
        assert_eq!(
            *message_map
                .get("data")
                .unwrap()
                .get_message()
                .get_column_as_vec_str("values")
                .first()
                .unwrap(),
            json_str_1
        );
        assert_eq!(message_map.get("chat").unwrap().get_name(), "from_s2_on_d2");
        assert_eq!(message_map.get("chat").unwrap().get_publisher(), "s2");
        assert_eq!(message_map.get("chat").unwrap().get_subject(), "d2");
        assert_eq!(
            *message_map.get("chat").unwrap().get_update(),
            ArrowTablePublish::Extend {
                table_name: "d2".to_string()
            }
        );
        assert_eq!(
            *message_map
                .get("chat")
                .unwrap()
                .get_message()
                .get_column_as_vec_str("values")
                .first()
                .unwrap(),
            json_str_2
        );

        Ok(())
    }
}
