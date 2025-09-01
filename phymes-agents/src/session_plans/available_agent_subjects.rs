use std::fmt::Display;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use phymes_core::{metrics::HashMap, schemas::{available_subjects::{create_blob_batch, create_messages_record_batch, create_queries_batch, create_timestamp_str, AvailableSubjects, AvailableSubjectsTrait}, messages::MessagesBuilderTraitExt}, session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::{arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait}, arrow_table_publish::ArrowTablePublish}, task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait, ArrowIncomingMessageTrait, ArrowMessageBuilderTrait}};
use serde::{Deserialize, Serialize};

/// Check that one or more of the [AvailableMessagingPublishSubjects], one or more of the [AvailableMessageSubscribeSubjects],
/// and optionally one or more of the [AvailableAttachmentPublishSubjects] and [AvailableAttachmentsSubscribeSubjects]
/// are provided in the SessionContextBuilder
pub fn check_agent_subjects(subjects: &[String]) -> Result<()> {
    let mut has_messaging_publish = false;
    let mut has_message_subscribe = false;

    for subject in subjects.iter() {
        if let Ok(interface_subject) = AvailableinterfaceSubjects::from_str(subject, false) {
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
            [AvailableinterfaceSubjects::UserMessages, AvailableinterfaceSubjects::UserQueries], subjects);
    }
    if !has_message_subscribe {
        anyhow::bail!("At least one AvailableInterface Message and Subscribe subject {:? }must be provided. Provided subjects were {:?}.", 
            [AvailableinterfaceSubjects::AssistantMessages, AvailableinterfaceSubjects::ToolMessages, AvailableinterfaceSubjects::AggregatedMessages], subjects);
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

/// The session interface composed of a type and direction
#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
pub struct SessionInterface {
    pub mode: SessionInterfaceMode,
    pub direction: SessionInterfaceDirection,
}

pub trait MessagingPublishSubjectsTrait {
    fn to_incoming_message(&self, content: &str, session_context_name: &str) -> Result<ArrowIncomingMessage>;
}

/// The available subjects that the user can publish on from the messaging interface
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableinterfaceSubjects {
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

impl Display for AvailableinterfaceSubjects {
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

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
pub struct MessageInterface {
    pub content: Vec<String>,
    pub role: Vec<String>,
    pub timestamp: Vec<i64>,
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
pub struct AttachmentInterface {
    pub filename: Vec<String>, 
    pub bytes: Vec<Vec<u8>>, 
    pub extension: Vec<String>, 
    pub metadata: Vec<String>,
}

impl AvailableSubjectsTrait for AvailableinterfaceSubjects {
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

impl AvailableinterfaceSubjects {
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

    pub fn to_incoming_message(&self, message: Option<MessageInterface>, attachment: Option<AttachmentInterface>, session_name: &str) -> Result<ArrowIncomingMessage> {
        match self {
            AvailableinterfaceSubjects::UserMessages => {
                if message.is_none() {
                    return Err(anyhow!("Specify the `MessageInterfaceInput` before building the message."))
                }
                // Make the system prompt and add the user query
                let table = ArrowTableBuilder::new()
                    .with_name(self.to_string().as_str())
                    // .insert_system_template_str("You are a helpful assistant.").unwrap()
                    // .append_new_user_query_str(&message.unwrap().content, "user")?;
                    .append_new_user_query(create_messages_record_batch(
                        message.unwrap().role, 
                        message.unwrap().content, 
                        message.unwrap().timestamp)?)?
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
            AvailableinterfaceSubjects::UserQueries => {
                if message.is_none() {
                    return Err(anyhow!("Specify the `MessageInterfaceInput` before building the message."))
                }
                // Make the query prompt
                let mut query_vec = Vec::new();
                let mut query_ids = Vec::new()
                for (content, timestamp) in message.unwrap().content.into_iter().zip(message.unwrap().timestamp.into_iter()) {
                    if cfg!(feature = "hf_hub") {
                        // DM: note that the prompt for the query is specific to Qwen!
                        let query_embed_str = format!(
                            "{}{}",
                            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                            content
                        );
                        query_vec.push(query_embed_str);
                    } else {
                        query_vec.push(content);
                    }
                    query_ids.push(timestamp.to_string());
                }
                let batch = create_queries_batch(query_ids, query_vec)?;

                let table = ArrowTableBuilder::new()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])
                    .unwrap()
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
                if attachment.is_none() {
                    return Err(anyhow!("Specify the `AttachmentInterfaceInput` before building the message."))
                }
                let batch = create_blob_batch(
                    attachment.unwrap().filename, 
                    attachment.unwrap().extension, 
                    attachment.unwrap().bytes,
                    attachment.unwrap().metadata
                )?;
                let table = ArrowTableBuilder::new()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])
                    .unwrap()
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

    pub fn from_incoming_message(&self, message: &ArrowIncomingMessage) -> Result<(Option<MessageInterface>, Option<AttachmentInterface>)> {
        match self {
            Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => {
                let content = message.get_message().get_column_as_vec_nonprimitive::<String>("content")?;
                let role = message.get_message().get_column_as_vec_nonprimitive::<String>("role")?;
                let timestamp = message.get_message().get_column_as_vec_primitive::<i64>("timestamp")?;
                let message = MessageInterface { role, content, timestamp };
                Ok((Some(content), None))
            },
            Self::AssistantCsv
            | Self::AssistantImage
            | Self::AssistantScript => {
                let filename = message.get_message().get_column_as_vec_nonprimitive::<String>("filename")?;
                let extension = message.get_message().get_column_as_vec_nonprimitive::<String>("extension")?;
                let bytes = message.get_message().get_column_as_vec_nested_primitive::<u8>("bytes")?;
                let metadata = message.get_message().get_column_as_vec_nonprimitive::<String>("metadata")?;
                let attachment = AttachmentInterface {filename, extension, bytes, metadata};
                Ok((None, Some(attachment)))
            },
            _ => return Err(anyhow!("Cannot extract from an incoming message for a poublication subject.")),
        }
    }
}