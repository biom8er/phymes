use std::fmt::Display;

use anyhow::Result;
use arrow::array::RecordBatch;
use clap::ValueEnum;
use phymes_core::{
    BuildableTrait, BuilderTrait, Subject, SubjectBuilder, SubjectPlan, SubjectPlanBuilderTrait,
};
use serde::{Deserialize, Serialize};

use crate::{AvailableSchemaTrait, AvailableSubjects, AvailableSubjectsTrait};

/// Check that one or more of the [AvailableInterfaceSubjects] are provided in the [SessionContextBuilder]
///
/// [SessionContextBuilder]: crate::SessionContextBuilder
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
                || interface_subject == AvailableInterfaceSubjects::ToolMessages
            {
                has_message_subscribe = true;
            }
        }
    }

    if !has_messaging_publish {
        anyhow::bail!(
            "At least one AvailableInterface Message and Publish subject {:?} must be provided. Provided subjects were {:?}.",
            [
                AvailableInterfaceSubjects::UserMessages,
                AvailableInterfaceSubjects::UserQueries
            ],
            subjects
        );
    }
    if !has_message_subscribe {
        anyhow::bail!(
            "At least one AvailableInterface Message and Subscribe subject {:? }must be provided. Provided subjects were {:?}.",
            [
                AvailableInterfaceSubjects::AssistantMessages,
                AvailableInterfaceSubjects::ToolMessages,
                AvailableInterfaceSubjects::AggregatedMessages
            ],
            subjects
        );
    }

    Ok(())
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
    #[value(name = "UserJson")]
    UserJson,
    #[value(name = "UserObject")]
    UserObject,
    #[value(name = "AggregatedMessages")]
    AggregatedMessages,
    #[value(name = "AggregatedAttachments")]
    AggregatedAttachments,
    #[value(name = "AssistantMessages")]
    AssistantMessages,
    #[value(name = "ToolMessages")]
    ToolMessages,
    #[value(name = "AssistantImage")]
    AssistantImage,
    #[value(name = "AssistantCsv")]
    AssistantCsv,
    #[value(name = "AssistantJson")]
    AssistantJson,
    #[value(name = "AssistantScript")]
    AssistantScript,
    #[value(name = "AssistantObject")]
    AssistantObject,
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
            Self::UserJson => write!(f, "UserJson"),
            Self::UserObject => write!(f, "UserObject"),
            Self::AggregatedMessages => write!(f, "AggregatedMessages"),
            Self::AggregatedAttachments => write!(f, "AggregatedAttachments"),
            Self::AssistantMessages => write!(f, "AssistantMessages"),
            Self::ToolMessages => write!(f, "ToolMessages"),
            Self::AssistantImage => write!(f, "AssistantImage"),
            Self::AssistantCsv => write!(f, "AssistantCsv"),
            Self::AssistantJson => write!(f, "AssistantJson"),
            Self::AssistantScript => write!(f, "AssistantScript"),
            Self::AssistantObject => write!(f, "AssistantObject"),
        }
    }
}

impl AvailableSubjectsTrait for AvailableInterfaceSubjects {
    fn to_subject(&self, name: Option<&str>, batches: Option<Vec<RecordBatch>>) -> Result<Subject> {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        match self {
            Self::UserMessages
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => {
                AvailableSubjects::Messages.to_subject(Some(name.as_str()), batches)
            }
            Self::UserQueries => {
                AvailableSubjects::Queries.to_subject(Some(name.as_str()), batches)
            }
            Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AggregatedAttachments
            | Self::AssistantScript => {
                AvailableSubjects::Attachments.to_subject(Some(name.as_str()), batches)
            }
            Self::UserObject | Self::AssistantObject => {
                AvailableSubjects::ObjectStore.to_subject(Some(name.as_str()), batches)
            }
        }
    }

    fn to_subject_builder(&self, name: Option<&str>) -> SubjectBuilder {
        let name = match name {
            Some(name) => name.to_string(),
            None => self.to_string(),
        };
        match self {
            Self::UserMessages
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => {
                AvailableSubjects::Messages.to_subject_builder(Some(name.as_str()))
            }
            Self::UserQueries => AvailableSubjects::Queries.to_subject_builder(Some(name.as_str())),
            Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AggregatedAttachments
            | Self::AssistantScript => {
                AvailableSubjects::Attachments.to_subject_builder(Some(name.as_str()))
            }
            Self::UserObject | Self::AssistantObject => {
                AvailableSubjects::ObjectStore.to_subject_builder(Some(name.as_str()))
            }
        }
    }

    fn to_subject_plan(
        &self,
        name: Option<&str>,
        batches: Option<Vec<RecordBatch>>,
    ) -> Result<SubjectPlan> {
        let subject = self.to_subject(name, batches)?;
        SubjectPlan::get_builder().with_subject(subject).build()
    }
}

impl AvailableSchemaTrait for AvailableInterfaceSubjects {
    fn to_schema(&self) -> arrow::datatypes::SchemaRef {
        match self {
            Self::UserMessages
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::ToolMessages => AvailableSubjects::Messages.to_schema(),
            Self::UserQueries => AvailableSubjects::Queries.to_schema(),
            Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AggregatedAttachments
            | Self::AssistantScript => AvailableSubjects::Attachments.to_schema(),
            Self::UserObject | Self::AssistantObject => AvailableSubjects::ObjectStore.to_schema(),
        }
    }
}

impl AvailableInterfaceSubjects {
    /// Is the subject subscribed to by the session?
    pub fn is_session_subscription(&self) -> bool {
        match self {
            Self::UserMessages
            | Self::UserQueries
            | Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::UserObject
            | Self::AggregatedAttachments
            | Self::AggregatedMessages => false,
            Self::AssistantMessages
            | Self::AssistantScript
            | Self::ToolMessages
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AssistantObject => true,
        }
    }
    /// Is the subject published to by the session?
    pub fn is_session_publication(&self) -> bool {
        match self {
            Self::UserMessages
            | Self::UserQueries
            | Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::UserObject => true,
            Self::AssistantMessages
            | Self::AssistantScript
            | Self::ToolMessages
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AssistantObject
            | Self::AggregatedAttachments
            | Self::AggregatedMessages => false,
        }
    }
    /// Is the subject subscribed to by the frontend UI?
    pub fn is_frontend_subscription(&self) -> bool {
        match self {
            Self::UserMessages
            | Self::UserQueries
            | Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::UserObject
            | Self::AssistantMessages
            | Self::AssistantScript
            | Self::ToolMessages
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AssistantObject => false,
            Self::AggregatedAttachments | Self::AggregatedMessages => true,
        }
    }
    /// Is the subject published to by the frontend UI?
    pub fn is_frontend_publication(&self) -> bool {
        match self {
            Self::UserMessages
            | Self::UserQueries
            | Self::UserPdf
            | Self::UserAudio
            | Self::UserVideo
            | Self::UserImage
            | Self::UserScript
            | Self::UserCsv
            | Self::UserJson
            | Self::UserObject => true,
            Self::AggregatedAttachments
            | Self::AggregatedMessages
            | Self::AssistantMessages
            | Self::AssistantScript
            | Self::ToolMessages
            | Self::AssistantImage
            | Self::AssistantCsv
            | Self::AssistantJson
            | Self::AssistantObject => false,
        }
    }
}
