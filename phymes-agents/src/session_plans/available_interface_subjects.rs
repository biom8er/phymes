use std::fmt::Display;

use anyhow::{anyhow, Result};
use arrow::array::RecordBatch;
use clap::{Parser, ValueEnum};
use phymes_core::{metrics::HashMap, schemas::available_subjects::{create_chat_fields, AvailableSubjects, AvailableSubjectsTrait}, session::common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, table::{table::{Table, TableBuilderTrait, TableTrait}, table_publish::TablePublish}, task::message::{ArrowIncomingIPCMessage, IPCMessage, IPCMessageBuilder, ArrowIncomingMessageBuilderTrait, ArrowIncomingMessageTrait, MessageBuilderTrait}};
use phymes_data::candle_data::summary_config::DataFormat;
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
            if interface_subject == AvailableInterfaceSubjects::UserMessages {
                has_messaging_publish = true;
            }
            if interface_subject == AvailableInterfaceSubjects::AssistantMessages 
            || interface_subject == AvailableInterfaceSubjects::AggregatedMessages
            || interface_subject == AvailableInterfaceSubjects::ToolMessages {
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
pub fn create_incoming_message_map(messages: Vec<IPCMessage>) -> HashMap<String, IPCMessage> {
    let mut incoming_message_map = HashMap::<String, IPCMessage>::new();
    for message in messages {
        incoming_message_map.insert(message.get_name().to_string(), message);
    }
    incoming_message_map
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
    // #[value(name = "SessionMetrics")]
    // SessionMetrics,
    // #[value(name = "SessionSchema")]
    // SessionMetricsAsGantt,
    // #[value(name = "SessionSchema")]
    // SessionSchema,
    // #[value(name = "SessionSchemaWithRows")]
    // SessionSchemaWithRows,

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

impl AvailableSubjectsTrait for AvailableInterfaceSubjects {
    fn to_table(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Table> {
        match self {
            Self::UserMessages 
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => AvailableSubjects::Messages.to_table(name, batches),
            Self::UserQueries => AvailableSubjects::Queries.to_table(name, batches),
            Self::UserPdf 
            | Self::UserAudio 
            | Self::UserVideo
            | Self::UserImage 
            | Self::UserScript 
            | Self::UserCsv 
            | Self::AssistantImage 
            | Self::AssistantCsv
            | Self::AssistantScript => AvailableSubjects::Blob.to_table(name, batches),
        }        
    }
    fn to_table_from_struct<T>(&self, name: Option<&str>, s: &[T]) -> Result<Table> where T: Sized + Serialize {
        match self {
            Self::UserMessages 
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => AvailableSubjects::Messages.to_table_from_struct::<T>(name, s),
            Self::UserQueries => AvailableSubjects::Queries.to_table_from_struct::<T>(name, s),
            Self::UserPdf 
            | Self::UserAudio 
            | Self::UserVideo
            | Self::UserImage 
            | Self::UserScript 
            | Self::UserCsv 
            | Self::AssistantImage 
            | Self::AssistantCsv
            | Self::AssistantScript => AvailableSubjects::Blob.to_table_from_struct::<T>(name, s),
        } 
    }
}