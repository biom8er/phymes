use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::{session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::{DataFormat, TablePublish}, task::{MessageBuilderTrait, MessageTrait}};

/// Composition of [MessageTrait] with additional functions for inter-session communication
pub trait SessionInterfaceMessageTrait: MessageTrait {
    fn get_session_name(&self) -> &str;
    fn get_format(&self) -> &DataFormat;
    fn get_stream(&self) -> &bool;    
}

/// Message format that can be communicated between different sessions
///   with different data formats
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct SessionInterfaceMessage {
    /// Name of the message
    name: String,
    /// The name of the subject
    subject: String,
    /// The name of the publishing task
    publisher: String,
    /// The actual message as byte stream
    message: Vec<u8>,
    /// How to update the state
    update: TablePublish,
    /// The name of the session
    session_name: String,
    /// Format of the message
    format: DataFormat,
    /// Stream the response
    stream: bool,
}

impl SessionInterfaceMessage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        subject: &str,
        publisher: &str,
        message: Option<Vec<u8>>,
        update: Option<TablePublish>,
        session_name: &str,
        format: &DataFormat,
        stream: bool
    ) -> Self {
        Self {
            name: name.to_string(),
            subject: subject.to_string(),
            publisher: publisher.to_string(),
            message: message.unwrap_or_default(),
            update: update.unwrap_or_default(),
            session_name: session_name.to_string(),
            format: format.to_owned(),
            stream
        }
    }
}

impl MappableTrait for SessionInterfaceMessage {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for SessionInterfaceMessage {
    type T = SessionInterfaceMessageBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl MessageTrait for SessionInterfaceMessage {
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

impl SessionInterfaceMessageTrait for SessionInterfaceMessage {
    fn get_session_name(&self) -> &str {
        &self.session_name
    }
    fn get_format(&self) -> &DataFormat {
        &self.format
    }
    fn get_stream(&self) -> &bool {
        &self.stream
    }
}

pub trait SessionInterfaceMessageBuilderTrait: MessageBuilderTrait {
    fn with_session_name(self, session_name: &str) -> Self;
    fn with_format(self, format: &DataFormat) -> Self;
    fn with_stream(self, stream: bool) -> Self;
}

#[derive(Default, Clone, PartialEq)]
pub struct SessionInterfaceMessageBuilder {
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
    /// The name of the session
    pub session_name: Option<String>,
    /// Format of the message
    pub format: Option<DataFormat>,
    /// Stream the response
    pub stream: Option<bool>,
}

impl BuilderTrait for SessionInterfaceMessageBuilder {
    type T = SessionInterfaceMessage;
    fn new() -> Self {
        Self {
            name: None,
            subject: None,
            publisher: None,
            message: None,
            update: None,
            session_name: None,
            format: None,
            stream: None,
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
            message: self.message.unwrap_or_default(),
            update: self.update.unwrap(),
            session_name: self.session_name.unwrap_or_default(),
            format: self.format.unwrap(),
            stream: self.stream.unwrap_or_default(),
        })
    }
}

impl MessageBuilderTrait for SessionInterfaceMessageBuilder {
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
        let session_name = match self.session_name {
            Some(ref s) => s,
            None => return Err(anyhow!("Cannot make name without session name")),
        };
        let name = format!("from_{publisher}_on_{subject}_in_{session_name}");
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
}

impl SessionInterfaceMessageBuilderTrait for SessionInterfaceMessageBuilder {
    fn with_session_name(mut self, session_name: &str) -> Self {
        self.session_name = Some(session_name.to_string());
        self
    }
    fn with_format(mut self, format: &DataFormat) -> Self {
        self.format = Some(*format);
        self
    }
    fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}