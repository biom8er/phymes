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

/// An Arrow Message
///
/// Contains information about what task published the message,
/// the subject, how to update the stream, and the message itself
pub trait ArrowMessageTrait: MappableTrait + BuildableTrait + Send {
    fn get_subject(&self) -> &str;
    fn get_publisher(&self) -> &str;
    fn get_update(&self) -> &ArrowTablePublish;
}

pub trait ArrowIncomingMessageTrait: ArrowMessageTrait + Sync {
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
    /// The actual message
    message: ArrowTable,
    /// How to update the state
    update: ArrowTablePublish,
}

impl ArrowIncomingMessage {
    pub fn new(
        name: &str,
        subject: &str,
        publisher: &str,
        message: Option<ArrowTable>,
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

        if self.message.get_schema().fields().contains(&fields) {
            // Each row is a new message
            let data = field_names
                .iter()
                .map(|f| self.message.get_column_as_str_vec(f))
                .collect::<Vec<_>>();
            let n_rows: usize = self
                .message
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
                let table = ArrowTableBuilder::new()
                    .with_name(name)
                    .with_record_batches(vec![batch])?
                    .build()?;
                let message = ArrowIncomingMessageBuilder::new()
                    // .with_name(name)
                    .with_publisher(data.get(1).unwrap().get(row).unwrap())
                    .with_subject(data.get(2).unwrap().get(row).unwrap())
                    .with_update(&ArrowTablePublish::Extend {
                        table_name: data.get(2).unwrap().get(row).unwrap().to_string(),
                    })
                    .with_message(table)
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
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &ArrowTablePublish {
        &self.update
    }
}

impl ArrowIncomingMessageTrait for ArrowIncomingMessage {
    fn get_message(&self) -> &ArrowTable {
        &self.message
    }
    fn get_message_own(self) -> ArrowTable {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut ArrowTable {
        &mut self.message
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArrowIncomingIPCMessage {
    /// Name of the message
    name: String,
    /// The name of the intended subject task
    subject: String,
    /// The name of the publisher task
    publisher: String,
    /// The actual message
    message: Vec<u8>,
    /// How to update the state
    update: ArrowTablePublish,
}

impl MappableTrait for ArrowIncomingIPCMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for ArrowIncomingIPCMessage {
    type T = ArrowIncomingIPCMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl ArrowMessageTrait for ArrowIncomingIPCMessage {
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &ArrowTablePublish {
        &self.update
    }
}

impl ArrowIncomingIPCMessageTrait for ArrowIncomingIPCMessage {
    fn get_message(&self) -> &Vec<u8> {
        &self.message
    }
    fn get_message_own(self) -> Vec<u8> {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut Vec<u8> {
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
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &ArrowTablePublish {
        &self.update
    }
}

impl ArrowOutgoingMessageTrait for ArrowOutgoingMessage {
    fn get_message(&self) -> &SendableRecordBatchStream {
        &self.message
    }
    fn get_message_own(self) -> SendableRecordBatchStream {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut SendableRecordBatchStream {
        &mut self.message
    }
}

pub struct ArrowOutgoingIPCMessage {
    /// Name of the message
    name: String,
    /// The name of the intended subject task
    subject: String,
    /// The name of the publisher task
    publisher: String,
    /// The actual message
    message: SendableIPCRecordBatchStream,
    /// How to update the state
    update: ArrowTablePublish,
}

impl MappableTrait for ArrowOutgoingIPCMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for ArrowOutgoingIPCMessage {
    type T = ArrowOutgoingIPCMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl ArrowMessageTrait for ArrowOutgoingIPCMessage {
    fn get_subject(&self) -> &str {
        &self.subject
    }
    fn get_publisher(&self) -> &str {
        &self.publisher
    }
    fn get_update(&self) -> &ArrowTablePublish {
        &self.update
    }
}

impl ArrowOutgoingIPCMessageTrait for ArrowOutgoingIPCMessage {
    fn get_message(&self) -> &SendableIPCRecordBatchStream {
        &self.message
    }
    fn get_message_own(self) -> SendableIPCRecordBatchStream {
        self.message
    }
    fn get_message_mut(&mut self) -> &mut SendableIPCRecordBatchStream {
        &mut self.message
    }
}

pub trait ArrowMessageBuilderTrait: BuilderTrait + Send {
    fn with_subject(self, name: &str) -> Self;
    fn with_publisher(self, name: &str) -> Self;
    fn make_name(self) -> Result<Self>
    where
        Self: Sized;
    fn with_update(self, update: &ArrowTablePublish) -> Self;
}

pub trait ArrowIncomingMessageBuilderTrait: ArrowMessageBuilderTrait + Sync {
    fn with_message(self, message: ArrowTable) -> Self;
}

pub trait ArrowOutgoingMessageBuilderTrait: ArrowMessageBuilderTrait {
    fn with_message(self, message: SendableRecordBatchStream) -> Self;
}

pub trait ArrowIPCMessageBuilderTrait: ArrowMessageBuilderTrait + Sync {
    fn with_message(self, message: Vec<u8>) -> Self;
}
pub trait ArrowOutgoingIPCMessageBuilderTrait: ArrowMessageBuilderTrait {
    fn with_message(self, message: SendableIPCRecordBatchStream) -> Self;
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
    pub message: Option<ArrowTable>,
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
}

impl ArrowIncomingMessageBuilderTrait for ArrowIncomingMessageBuilder {
    fn with_message(mut self, message: ArrowTable) -> Self {
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

impl ArrowOutgoingMessageBuilder {
    pub fn new_from_outgoing_ipc_message(message: ArrowOutgoingIPCMessage) -> Self {
        Self {
            name: Some(message.get_name().to_string()),
            subject: Some(message.get_subject().to_string()),
            publisher: Some(message.get_publisher().to_string()),
            update: Some(message.get_update().clone()),
            message: Some(Box::pin(ArrowMessageStreamDeserializer {
                schema: message.get_message().schema(),
                input: message.get_message_own(),
            })),
        }
    }
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
}

impl ArrowOutgoingMessageBuilderTrait for ArrowOutgoingMessageBuilder {
    fn with_message(mut self, message: SendableRecordBatchStream) -> Self {
        self.message = Some(message);
        self
    }
}

#[derive(Default, Clone)]
pub struct ArrowIncomingIPCMessageBuilder {
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

impl BuilderTrait for ArrowIncomingIPCMessageBuilder {
    type T = ArrowIncomingIPCMessage;
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

impl ArrowMessageBuilderTrait for ArrowIncomingIPCMessageBuilder {
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
}

impl ArrowIPCMessageBuilderTrait for ArrowIncomingIPCMessageBuilder {
    fn with_message(mut self, message: Vec<u8>) -> Self {
        self.message = Some(message);
        self
    }
}

#[derive(Default)]
pub struct ArrowOutgoingIPCMessageBuilder {
    /// Name of the message
    pub name: Option<String>,
    /// The name of the intended subject task
    pub subject: Option<String>,
    /// The name of the publisher task
    pub publisher: Option<String>,
    /// The actually message
    pub message: Option<SendableIPCRecordBatchStream>,
    /// How to update the state
    pub update: Option<ArrowTablePublish>,
}

impl ArrowOutgoingIPCMessageBuilder {
    pub fn new_from_outgoing_message(message: ArrowOutgoingMessage) -> Self {
        Self {
            name: Some(message.get_name().to_string()),
            subject: Some(message.get_subject().to_string()),
            publisher: Some(message.get_publisher().to_string()),
            update: Some(message.get_update().clone()),
            message: Some(Box::pin(ArrowMessageStreamSerializer {
                schema: message.get_message().schema(),
                input: message.get_message_own(),
            })),
        }
    }
}

impl BuilderTrait for ArrowOutgoingIPCMessageBuilder {
    type T = ArrowOutgoingIPCMessage;
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

impl ArrowMessageBuilderTrait for ArrowOutgoingIPCMessageBuilder {
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
}

impl ArrowOutgoingIPCMessageBuilderTrait for ArrowOutgoingIPCMessageBuilder {
    fn with_message(mut self, message: SendableIPCRecordBatchStream) -> Self {
        self.message = Some(message);
        self
    }
}

struct ArrowMessageStreamDeserializer {
    /// The schema of the stream
    schema: SchemaRef,
    /// The input task to process.
    input: SendableIPCRecordBatchStream,
}

fn deserialize_batches(
    bytes: Vec<u8>,
    // could also be other arguments required for processing
) -> Result<RecordBatch> {
    let table = ArrowTableBuilder::new_from_ipc_stream(&bytes)?
        .with_name("tmp")
        .build()?;
    Ok(table.get_record_batches().first().unwrap().to_owned())
}

impl Stream for ArrowMessageStreamDeserializer {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match ready!(self.input.poll_next_unpin(cx)) {
            Some(Ok(bytes)) => {
                let processed_batch = deserialize_batches(bytes)?;
                Poll::Ready(Some(Ok(processed_batch)))
            }
            _ => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Same number of record batches
        self.input.size_hint()
    }
}

impl RecordBatchStream for ArrowMessageStreamDeserializer {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
struct ArrowMessageStreamSerializer {
    /// The schema of the stream
    schema: SchemaRef,
    /// The input task to process.
    input: SendableRecordBatchStream,
}

fn serialize_batches(
    batch: RecordBatch,
    // could also be other arguments required for processing
) -> Result<Vec<u8>> {
    let table = ArrowTableBuilder::new()
        .with_name("tmp")
        .with_record_batches(vec![batch])?
        .build()?;
    table.to_ipc_stream()
}

impl Stream for ArrowMessageStreamSerializer {
    type Item = Result<Vec<u8>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match ready!(self.input.poll_next_unpin(cx)) {
            Some(Ok(batch)) => {
                let processed_batch = serialize_batches(batch)?;
                Poll::Ready(Some(Ok(processed_batch)))
            }
            _ => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Same number of record batches
        self.input.size_hint()
    }
}

impl IPCRecordBatchStream for ArrowMessageStreamSerializer {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
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

    #[tokio::test]
    async fn test_convert_ipc_to_outgoing_message() -> Result<()> {
        let test_table = test_table::make_test_table("test_table", 4, 8, 3)?;

        // Make the Outgoing Message
        let outgoing_message = ArrowOutgoingMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&ArrowTablePublish::None)
            .with_message(test_table.clone().to_record_batch_stream())
            .build()?;

        // Convert to an IPC stream then back to outgoing message
        let outgoing_ipc_message =
            ArrowOutgoingIPCMessageBuilder::new_from_outgoing_message(outgoing_message).build()?;
        let outgoing_message =
            ArrowOutgoingMessageBuilder::new_from_outgoing_ipc_message(outgoing_ipc_message)
                .build()?;
        assert_eq!(outgoing_message.get_name(), "name");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(*outgoing_message.get_update(), ArrowTablePublish::None);

        // Get back the original table
        let test_table_read = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            outgoing_message.get_message_own(),
        )
        .await?
        .with_name("test_table")
        .build()?;

        assert_eq!(test_table.get_name(), test_table_read.get_name());
        assert_eq!(test_table.get_schema(), test_table_read.get_schema());
        assert_eq!(
            test_table.get_record_batches(),
            test_table_read.get_record_batches()
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
                .get_column_as_str_vec("values")
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
                .get_column_as_str_vec("values")
                .first()
                .unwrap(),
            json_str_2
        );

        Ok(())
    }
}
