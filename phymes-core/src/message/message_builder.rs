use crate::{
    BuilderTrait, IPCMessage, Publication, SendableRecordBatchStream,
    SendableRecordBatchStreamMessage,
};

use anyhow::{Result, anyhow};

/// Utility function to create a random ID
pub fn make_random_id() -> Result<u128> {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).map_err(|e| anyhow!("{e:?}"))?;
    let hash = u128::from_ne_bytes(buf);
    Ok(hash)
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
    fn with_update(self, update: &Publication) -> Self;
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
    pub update: Option<Publication>,
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
    fn with_update(mut self, update: &Publication) -> Self {
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
        let hash = make_random_id()?;
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
        if self.update.as_ref().unwrap() != &Publication::None
            && self.subject.as_ref().unwrap() != self.update.as_ref().unwrap().subject_name()
        {
            Err(anyhow!(
                "Mismatch between provided subject {} and table publish table name {}.",
                self.subject.as_ref().unwrap(),
                self.update.as_ref().unwrap().subject_name()
            ))
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
    pub update: Option<Publication>,
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
    fn with_update(mut self, update: &Publication) -> Self {
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
        let hash = make_random_id()?;
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
        if self.update.as_ref().unwrap() != &Publication::None
            && self.subject.as_ref().unwrap() != self.update.as_ref().unwrap().subject_name()
        {
            Err(anyhow!(
                "Mismatch between provided subject {} and table publish table name {}.",
                self.subject.as_ref().unwrap(),
                self.update.as_ref().unwrap().subject_name()
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{MappableTrait, MessageTrait, SubjectTrait, test_subject};

    use super::*;

    #[test]
    fn test_arrow_message_buiilders_success() -> Result<()> {
        // Test data
        let test_table = test_subject::make_test_subject("test_table", 4, 8, 3)?;

        // Case 1: with name
        let incoming_message = IPCMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&Publication::Extend {
                subject_name: "subject".to_string(),
            })
            .with_message(test_table.to_ipc_stream()?)
            .build()?;
        assert_eq!(incoming_message.get_name(), "name");
        assert_eq!(incoming_message.get_subject(), "subject");
        assert_eq!(incoming_message.get_publisher(), "publisher");
        assert_eq!(
            *incoming_message.get_update(),
            Publication::Extend {
                subject_name: "subject".to_string()
            }
        );

        let outgoing_message = SendableRecordBatchStreamMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&Publication::Extend {
                subject_name: "subject".to_string(),
            })
            .with_message(test_table.to_record_batch_stream())
            .build()?;
        assert_eq!(outgoing_message.get_name(), "name");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(
            *outgoing_message.get_update(),
            Publication::Extend {
                subject_name: "subject".to_string()
            }
        );
        assert_eq!(
            outgoing_message.get_message().schema(),
            test_table.get_schema()
        );

        // Case 2: make name
        let incoming_message = IPCMessageBuilder::new()
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&Publication::None)
            .make_name()?
            .with_message(test_table.to_ipc_stream()?)
            .build()?;
        assert_eq!(incoming_message.get_name(), "from_publisher_on_subject");
        assert_eq!(incoming_message.get_subject(), "subject");
        assert_eq!(incoming_message.get_publisher(), "publisher");
        assert_eq!(*incoming_message.get_update(), Publication::None);

        let outgoing_message = SendableRecordBatchStreamMessageBuilder::new()
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&Publication::None)
            .make_name()?
            .with_message(test_table.to_record_batch_stream())
            .build()?;
        assert_eq!(outgoing_message.get_name(), "from_publisher_on_subject");
        assert_eq!(outgoing_message.get_subject(), "subject");
        assert_eq!(outgoing_message.get_publisher(), "publisher");
        assert_eq!(*outgoing_message.get_update(), Publication::None);
        assert_eq!(
            outgoing_message.get_message().schema(),
            test_table.get_schema()
        );

        Ok(())
    }

    #[test]
    fn test_arrow_message_buiilders_mismatched_subjects() -> Result<()> {
        // Test data
        let test_table = test_subject::make_test_subject("test_table", 4, 8, 3)?;

        // Case 1: with name
        let result = IPCMessageBuilder::new()
            .with_name("name")
            .with_subject("subject")
            .with_publisher("publisher")
            .with_update(&Publication::Extend {
                subject_name: "mismatch".to_string(),
            })
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
            .with_update(&Publication::Extend {
                subject_name: "mismatch".to_string(),
            })
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
}
