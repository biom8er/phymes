use anyhow::Result;
use clap::ValueEnum;
use phymes_core::{schemas::{available_subjects::{create_blob_batch, create_queries_batch, create_timestamp_str, AvailableSubjects}, messages::MessagesBuilderTraitExt}, session::common_traits::{BuilderTrait, MappableTrait}, table::{arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait}, arrow_table_publish::ArrowTablePublish}, task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait, ArrowIncomingMessageTrait, ArrowMessageBuilderTrait}};
use serde::{Deserialize, Serialize};

/// Check that one or more of the [AvailableMessagingPublishSubjects], one or more of the [AvailableMessageSubscribeSubjects],
/// and optionally one or more of the [AvailableAttachmentPublishSubjects] and [AvailableAttachmentsSubscribeSubjects]
/// are provided in the SessionContextBuilder
pub fn check_agent_subjects(subjects: &[String]) -> Result<()> {
    let mut has_messaging_publish = false;
    let mut has_message_subscribe = false;

    for subject in subjects.iter() {
        if AvailableMessagingPublishSubjects::from_str(subject, false).is_ok() {
            has_messaging_publish = true;
        }
        if AvailableMessageSubscribeSubjects::from_str(subject, false).is_ok() {
            has_message_subscribe = true;
        }
    }

    if !has_messaging_publish {
        anyhow::bail!("At least one AvailableMessagingPublishSubject must be provided");
    }
    if !has_message_subscribe {
        anyhow::bail!("At least one AvailableMessageSubscribeSubject must be provided");
    }

    Ok(())
}

pub trait AvailableSubjectsTrait {
    fn to_table(&self) -> Result<ArrowTable>;
}

pub trait MessagingPublishSubjectsTrait {
    fn to_incoming_message(&self, content: &str, session_context_name: &str) -> Result<ArrowIncomingMessage>;
}

/// The available subjects that the user can publish on from the messaging interface
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableMessagingPublishSubjects {
    #[default]
    UserMessages,
    UserQueries,
}

impl AvailableSubjectsTrait for AvailableMessagingPublishSubjects {
    fn to_table(&self) -> Result<ArrowTable> {
        AvailableSubjects::Messages.to_table(self.get_name())
    }
}

impl MappableTrait for AvailableMessagingPublishSubjects {
    fn get_name(&self) -> &str {
        match self {
            AvailableMessagingPublishSubjects::UserMessages => "UserMessages",
            AvailableMessagingPublishSubjects::UserQueries => "UserQueries",
        }
    }
}

impl MessagingPublishSubjectsTrait for AvailableMessagingPublishSubjects {
    fn to_incoming_message(&self, content: &str, session_context_name: &str) -> Result<ArrowIncomingMessage> {
        match self {
            AvailableMessagingPublishSubjects::UserMessages => {
                // Make the system prompt and add the user query
                let message_builder = ArrowTableBuilder::new()
                    .with_name(self.get_name())
                    // .insert_system_template_str("You are a helpful assistant.").unwrap()
                    .append_new_user_query_str(content, "user")?;

                // Build the current message state
                ArrowIncomingMessageBuilder::new()
                    .with_name(self.get_name())
                    .with_subject(self.get_name())
                    .with_publisher(session_context_name)
                    .with_message(message_builder.clone().build()?)
                    .with_update(&ArrowTablePublish::Extend {
                        table_name: self.get_name().to_string(),
                    })
                    .build()
            }
            AvailableMessagingPublishSubjects::UserQueries => {
                // Make the query prompt
                let mut query_vec = Vec::new();
                if cfg!(feature = "hf_hub") {
                    // DM: note that the prompt for the query is specific to Qwen!
                    let query_embed_str = format!(
                        "{}{}",
                        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                        content
                    );
                    query_vec.push(query_embed_str);
                } else {
                    query_vec.push(content.to_string());
                }
                let batch = create_queries_batch(query_vec, vec![create_timestamp_str()])?;

                let table = ArrowTableBuilder::new()
                    .with_name(self.get_name())
                    .with_record_batches(vec![batch])
                    .unwrap()
                    .build()?;

                ArrowIncomingMessageBuilder::new()
                    .with_name(self.get_name())
                    .with_subject(self.get_name())
                    .with_publisher(session_context_name)
                    .with_message(table)
                    .with_update(&ArrowTablePublish::Replace {
                        table_name: self.get_name().to_string(),
                    })
                    .build()
            }
        }
    }
}
pub trait AttachmentPublishSubjectsTrait {
    fn to_incoming_message(&self, filename: &str, bytes: &[u8], extension: &str, metadata: &str, session_context_name: &str) -> Result<ArrowIncomingMessage>;
}

/// The available subjects that the user can publish via attachments
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableAttachmentPublishSubjects {
    #[default]
    UserPdf,
    UserAudio,
    UserVideo,
    UserImage,
    UserScript,
}

impl AvailableSubjectsTrait for AvailableAttachmentPublishSubjects {
    fn to_table(&self) -> Result<ArrowTable> {
        AvailableSubjects::Blobs.to_table(self.get_name())
    }
}

impl MappableTrait for AvailableAttachmentPublishSubjects {
    fn get_name(&self) -> &str {
        match self {
            AvailableAttachmentPublishSubjects::UserPdf => "UserPdf",
            AvailableAttachmentPublishSubjects::UserAudio => "UserAudio",
            AvailableAttachmentPublishSubjects::UserVideo => "UserVideo",
            AvailableAttachmentPublishSubjects::UserImage => "UserImage",
            AvailableAttachmentPublishSubjects::UserScript => "UserScript",
        }
    }
}

impl AttachmentPublishSubjectsTrait for AvailableAttachmentPublishSubjects {
    fn to_incoming_message(&self, filename: &str, bytes: &[u8], extension: &str, metadata: &str, session_context_name: &str) -> Result<ArrowIncomingMessage> {
        let batch = create_blob_batch(vec![filename.to_string()], vec![extension.to_string()], vec![bytes.to_vec()], vec![metadata.to_string()])?;
        let table = ArrowTableBuilder::new()
            .with_name(self.get_name())
            .with_record_batches(vec![batch])
            .unwrap()
            .build()?;

        ArrowIncomingMessageBuilder::new()
            .with_name(self.get_name())
            .with_subject(self.get_name())
            .with_publisher(session_context_name)
            .with_message(table)
            .with_update(&ArrowTablePublish::Extend {
                table_name: self.get_name().to_string(),
            })
            .build()
    }
}

pub trait MessageSubscribeSubjectsTrait {
    fn from_incomeing_message(&self, message: &ArrowIncomingMessage) -> Result<String>;
}

/// The available subjects that the user can subscribe to and view in the messaging interface
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableMessageSubscribeSubjects {
    #[default]
    AssistantMessages,
    ToolMessages,
}

impl AvailableSubjectsTrait for AvailableMessageSubscribeSubjects {
    fn to_table(&self) -> Result<ArrowTable> {
        AvailableSubjects::Messages.to_table(self.get_name())
    }
}

impl MappableTrait for AvailableMessageSubscribeSubjects {
    fn get_name(&self) -> &str {
        match self {
            AvailableMessageSubscribeSubjects::AssistantMessages => "AssistantMessages",
            AvailableMessageSubscribeSubjects::ToolMessages => "ToolMessages",
        }
    }
}

impl MessageSubscribeSubjectsTrait for AvailableMessageSubscribeSubjects {
    fn from_incomeing_message(&self, message: &ArrowIncomingMessage) -> Result<String> {
        // let filenames = message.get_message_own().get_column_as_vec_nonprimitive::<String>("filename")?;
        let content_vec = message.get_message().get_column_as_vec_str("content");
        let content = content_vec.join("");
        Ok(content)
    }
}

pub trait AttachmentSubscribeSubjectsTrait {
    fn from_incomeing_message(&self, message: &ArrowIncomingMessage) -> Result<(Vec<String>, Vec<String>, Vec<Vec<u8>>)>;
}

/// The available subjects that the user can subscribe to via attachments
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableAttachmentsSubscribeSubjects {
    #[default]
    AssistantImage,
    AssistantScript,
}

impl AvailableSubjectsTrait for AvailableAttachmentsSubscribeSubjects {
    fn to_table(&self) -> Result<ArrowTable> {
        AvailableSubjects::Blobs.to_table(self.get_name())
    }
}

impl MappableTrait for AvailableAttachmentsSubscribeSubjects {
    fn get_name(&self) -> &str {
        match self {
            AvailableAttachmentsSubscribeSubjects::AssistantImage => "AssistantImage",
            AvailableAttachmentsSubscribeSubjects::AssistantScript => "AssistantScript",
        }
    }
}

impl AttachmentSubscribeSubjectsTrait for AvailableAttachmentsSubscribeSubjects {
    fn from_incomeing_message(&self, message: &ArrowIncomingMessage) -> Result<(Vec<String>, Vec<String>, Vec<Vec<u8>>)> {
        let filenames = message.get_message().get_column_as_vec_nonprimitive::<String>("filename")?;
        let extensions = message.get_message().get_column_as_vec_nonprimitive::<String>("extension")?;
        let bytes_vec = message.get_message().get_column_as_vec_nested_primitive::<u8>("bytes")?;
        Ok((filenames, extensions, bytes_vec))
    }
}