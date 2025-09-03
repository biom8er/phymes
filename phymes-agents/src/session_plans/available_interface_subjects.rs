use std::fmt::Display;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use phymes_core::{metrics::HashMap, schemas::{available_subjects::{AvailableSubjects, AvailableSubjectsTrait}}, session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::{arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait}, arrow_table_publish::ArrowTablePublish}, task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait, ArrowIncomingMessageTrait, ArrowMessageBuilderTrait}};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Check that one or more of the [AvailableinterfaceSubjects], one or more of the [AvailableinterfaceSubjects],
/// and optionally one or more of the [AvailableinterfaceSubjects] and [AvailableinterfaceSubjects]
/// are provided in the SessionContextBuilder
pub fn check_agent_subjects(subjects: &[String]) -> Result<()> {
    let mut has_messaging_publish = false;
    let mut has_message_subscribe = false;

    for subject in subjects.iter() {
        if let Ok(interface_subject) = AvailableInterfaceSubjects::from_str(subject, false) {
            if interface_subject.get_mode() == SessionInterfaceMode::Message && interface_subject.get_direction() == SessionInterfaceDirection::Publish {
                has_messaging_publish = true;
            }
            if interface_subject.get_mode() == SessionInterfaceMode::Message && interface_subject.get_direction() == SessionInterfaceDirection::Subscribe {
                has_message_subscribe = true;
            }
        }
    }

    if !has_messaging_publish {
        anyhow::bail!("At least one AvailableInterface Message and Publish subject {:?} must be provided. Provided subjects were {:?}.", 
            [AvailableInterfaceSubjects::UserMessages, AvailableInterfaceSubjects::UserQueries], subjects);
    }
    if !has_message_subscribe {
        anyhow::bail!("At least one AvailableInterface Message and Subscribe subject {:? }must be provided. Provided subjects were {:?}.", 
            [AvailableInterfaceSubjects::AssistantMessages, AvailableInterfaceSubjects::ToolMessages, AvailableInterfaceSubjects::AggregatedMessages], subjects);
    }

    Ok(())
}

/// Helper function to create the incoming message map from a vector of incoming messages
pub fn create_incoming_message_map(messages: Vec<ArrowIncomingMessage>) -> HashMap<String, ArrowIncomingMessage> {
    let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
    for message in messages {
        incoming_message_map.insert(message.get_name().to_string(), message);
    }
    incoming_message_map
}

/// Session interface mode: Message (text) or Attachment (bytes)
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum SessionInterfaceMode {
    #[default]    
    #[value(name = "Message")]
    Message,
    #[value(name = "Attachment")]
    Attachment
}

/// Session interface direction: Publish or Subscribe
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum SessionInterfaceDirection {
    #[default]    
    #[value(name = "Publish")]
    Publish,
    #[value(name = "Subscribe")]
    Subscribe
}

/// The available subjects that the user can publish on from the messaging interface
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableInterfaceSubjects {
    #[default]    
    #[value(name = "UserMessages")]
    UserMessages,
    #[value(name = "UserQueries")]
    UserQueries,
    #[value(name = "UserPdf")]
    UserPdf,
    #[value(name = "UserAudio")]
    UserAudio,
    #[value(name = "UserVideo")]
    UserVideo,
    #[value(name = "UserImage")]
    UserImage,
    #[value(name = "UserScript")]
    UserScript,
    #[value(name = "UserCsv")]
    UserCsv,
    #[value(name = "AggregatedMessages")]
    AggregatedMessages,
    #[value(name = "AssistantMessages")]
    AssistantMessages,
    #[value(name = "ToolMessages")]
    ToolMessages,
    #[value(name = "AssistantImage")]
    AssistantImage,
    #[value(name = "AssistantCsv")]
    AssistantCsv,
    #[value(name = "AssistantScript")]
    AssistantScript,
}

impl Display for AvailableInterfaceSubjects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserMessages => write!(f, "UserMessages"),
            Self::UserQueries => write!(f, "UserQueries"),
            Self::UserPdf => write!(f, "UserPdf"),
            Self::UserAudio => write!(f, "UserAudio"),
            Self::UserVideo => write!(f, "UserVideo"),
            Self::UserImage => write!(f, "UserImage"),
            Self::UserScript => write!(f, "UserScript"),
            Self::UserCsv => write!(f, "UserCsv"),
            Self::AggregatedMessages => write!(f, "AggregatedMessages"),
            Self::AssistantMessages => write!(f, "AssistantMessages"),
            Self::ToolMessages => write!(f, "ToolMessages"),
            Self::AssistantImage => write!(f, "AssistantImage"),
            Self::AssistantCsv => write!(f, "AssistantCsv"),
            Self::AssistantScript => write!(f, "AssistantScript"),
        }
    }
}

/// Schema of the message
#[derive(Parser, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MessageInterface {
    pub content: String,
    pub role: String,
    pub timestamp: i64,
}

/// Schema of the attachment
#[derive(Parser, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AttachmentInterface {
    pub filename: String, 
    pub bytes: Vec<u8>, 
    pub extension: String, 
    pub metadata: String,
    pub timestamp: i64,
}

impl AvailableSubjectsTrait for AvailableInterfaceSubjects {
    fn to_table(&self, name: Option<&str>) -> Result<ArrowTable> {        
        match self {
            Self::UserMessages 
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => AvailableSubjects::Messages.to_table(name),
            Self::UserQueries => AvailableSubjects::Queries.to_table(name),
            Self::UserPdf 
            | Self::UserAudio 
            | Self::UserVideo
            | Self::UserImage 
            | Self::UserScript 
            | Self::UserCsv 
            | Self::AssistantImage 
            | Self::AssistantCsv
            | Self::AssistantScript => AvailableSubjects::Blob.to_table(name),
        }        
    }
}

impl AvailableInterfaceSubjects {
    /// Get the mode of the subject
    pub fn get_mode(&self) -> SessionInterfaceMode {
        match self {
            Self::UserMessages 
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages
            | Self::UserQueries => SessionInterfaceMode::Message,
            Self::UserPdf 
            | Self::UserAudio 
            | Self::UserVideo
            | Self::UserImage 
            | Self::UserScript 
            | Self::UserCsv 
            | Self::AssistantImage 
            | Self::AssistantCsv
            | Self::AssistantScript => SessionInterfaceMode::Attachment,
        }
    }

    /// Get the direction of the subject
    pub fn get_direction(&self) -> SessionInterfaceDirection {
        match self {
            Self::UserMessages 
            | Self::UserQueries
            | Self::UserPdf 
            | Self::UserAudio 
            | Self::UserVideo
            | Self::UserImage 
            | Self::UserScript 
            | Self::UserCsv  => SessionInterfaceDirection::Publish,
            Self::AssistantImage 
            | Self::AssistantCsv
            | Self::AssistantScript
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => SessionInterfaceDirection::Subscribe
        }
    }

    /// Create an incoming message from either a message or attachment
    pub fn to_incoming_message(&self, message: Option<Vec<MessageInterface>>, attachment: Option<Vec<AttachmentInterface>>, session_name: &str) -> Result<ArrowIncomingMessage> {
        match self {
            AvailableInterfaceSubjects::UserMessages => {
                // Extract out the messages
                if message.is_none() {
                    return Err(anyhow!("Specify the `MessageInterfaceInput` before building the message."))
                }
                let batch_size = message.iter().len();
                let bytes = serde_json::to_vec(&message.unwrap())?;
                let table = ArrowTable::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_schema(AvailableSubjects::Messages.to_schema())
                    .with_json(&bytes, batch_size)?
                    .build()?;

                // Build the current message state
                ArrowIncomingMessageBuilder::new()
                    .with_name(self.to_string().as_str())
                    .with_subject(self.to_string().as_str())
                    .with_publisher(session_name)
                    .with_message(table)
                    .with_update(&ArrowTablePublish::Extend {
                        table_name: self.to_string(),
                    })
                    .build()
            }
            AvailableInterfaceSubjects::UserQueries => {
                // Extract out the messages
                if message.is_none() {
                    return Err(anyhow!("Specify the `MessageInterfaceInput` before building the message."))
                }
                let queries = message.unwrap().into_iter()
                    .map(|m| {
                        let content = if cfg!(feature = "hf_hub") {
                            // DM: note that the prompt for the query is specific to Qwen!
                            format!(
                                "{}{}",
                                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                                m.content
                            )
                        } else {
                            m.content
                        };
                        let id = m.timestamp.to_string();
                        json!({"query_id": id, "text": content})
                    }).collect::<Vec<_>>();
                
                // Make the table
                let table = ArrowTable::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_schema(AvailableSubjects::Queries.to_schema())
                    .with_json_values(&queries)?
                    .build()?;

                ArrowIncomingMessageBuilder::new()
                    .with_name(self.to_string().as_str())
                    .with_subject(self.to_string().as_str())
                    .with_publisher(session_name)
                    .with_message(table)
                    .with_update(&ArrowTablePublish::Replace {
                        table_name: self.to_string(),
                    })
                    .build()
            },
            Self::UserPdf 
            | Self::UserAudio 
            | Self::UserVideo
            | Self::UserImage 
            | Self::UserScript 
            | Self::UserCsv  => {
                // Extract out the attachments
                if attachment.is_none() {
                    return Err(anyhow!("Specify the `AttachmentInterfaceInput` before building the message."))
                }
                let batch_size = attachment.iter().len();
                let bytes = serde_json::to_vec(&attachment.unwrap())?;
                let table = ArrowTable::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_schema(AvailableSubjects::Blob.to_schema())
                    .with_json(&bytes, batch_size)?
                    .build()?;

                ArrowIncomingMessageBuilder::new()
                    .with_name(self.to_string().as_str())
                    .with_subject(self.to_string().as_str())
                    .with_publisher(session_name)
                    .with_message(table)
                    .with_update(&ArrowTablePublish::Extend {
                        table_name: self.to_string().to_string(),
                    })
                    .build()
            }
            _ => return Err(anyhow!("Cannot build an incoming message for a subscription subject.")),
        }
    }

    /// Extract out the contents for display as a message or attachment
    pub fn from_incoming_message(&self, message: &ArrowIncomingMessage) -> Result<(Option<Vec<MessageInterface>>, Option<Vec<AttachmentInterface>>)> {
        match self {
            Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => {
                let role = message.get_message().get_column_as_vec_nonprimitive::<String>("role")?;
                let content = message.get_message().get_column_as_vec_nonprimitive::<String>("content")?;
                let timestamp = message.get_message().get_column_as_vec_primitive::<i64>("timestamp")?;
                let message = role.into_iter()
                    .zip(content.into_iter())
                    .zip(timestamp.into_iter())
                    .map(|((role, content), timestamp)| MessageInterface { role, content, timestamp })
                    .collect::<Vec<_>>();
                Ok((Some(message), None))
            },
            Self::AssistantCsv
            | Self::AssistantImage
            | Self::AssistantScript => {
                let filename = message.get_message().get_column_as_vec_nonprimitive::<String>("filename")?;
                let extension = message.get_message().get_column_as_vec_nonprimitive::<String>("extension")?;
                let bytes = message.get_message().get_column_as_vec_nested_primitive::<u8>("bytes")?;
                let metadata = message.get_message().get_column_as_vec_nonprimitive::<String>("metadata")?;
                let timestamp = message.get_message().get_column_as_vec_primitive::<i64>("timestamp")?;
                let attachment = filename.into_iter()
                    .zip(extension.into_iter())
                    .zip(bytes.into_iter())
                    .zip(metadata.into_iter())
                    .zip(timestamp.into_iter())
                    .map(|((((filename, extension), bytes), metadata), timestamp)| AttachmentInterface {filename, extension, bytes, metadata, timestamp})
                    .collect::<Vec<_>>();
                Ok((None, Some(attachment)))
            },
            _ => return Err(anyhow!("Cannot extract from an incoming message for a poublication subject.")),
        }
    }
}

/// Server session request
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct SessionInterface {
    /// The name of the session plan
    pub session_plan: String,
    /// The name of the session
    pub session_name: String,
    /// Message or Attachment
    pub mode: SessionInterfaceMode,
    /// Publish or Subscribe
    pub direction: SessionInterfaceDirection,
    /// The subject name
    pub subject_name: Option<String>,
    /// The message content
    pub messaging: Option<MessageInterface>,
    /// The attachment content
    pub attachment: Option<AttachmentInterface>,
    /// Stream the response
    pub stream: bool,
}