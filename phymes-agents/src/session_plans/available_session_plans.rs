use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use parking_lot::RwLock;
use phymes_core::{
    metrics::ArrowTaskMetricsSet,
    session::{
        common_traits::BuilderTrait,
        session_context::{SessionStream, SessionStreamState},
        session_context_builder::SessionContextBuilderTrait,
    },
};
use serde::{Deserialize, Serialize};

use crate::{session_plans::available_agent_subjects::{create_incoming_message_map, AttachmentInterface, AvailableinterfaceSubjects, MessageInterface}, session_traits::agents::{CustomAgentsBuilderTrait, SessionContextBuilderAgentsTrait}};

use super::{
    chat_agent_session::ChatAgentSession,
    document_rag_session::DocumentRAGSession,
    tool_agent_session::ToolAgentSession,
};

/// The available session plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum AvailableSessionPlans {
    #[value(name = "Chat")]
    Chat,
    #[value(name = "DocChat")]
    DocChat,
    #[value(name = "ToolChat")]
    ToolChat,
}

impl Display for AvailableSessionPlans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "Chat"),
            Self::DocChat => write!(f, "DocChat"),
            Self::ToolChat => write!(f, "ToolChat"),
        }
    }
}

impl AvailableSessionPlans {
    /// Get all available session plans
    pub fn get_all_session_plan_names() -> Vec<String> {
        let session_plans = ["Chat", "DocChat", "ToolChat"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get the session stream state
    pub fn get_session_stream_state(&self, session_name: &str) -> Arc<RwLock<SessionStreamState>> {
        // Initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // Initialize the session
        match self {
            Self::Chat => {
                let session = ChatAgentSession::new_with_session_name(session_name);
                let session_ctx = session
                    .build()
                    .with_metrics(metrics.clone())
                    .with_name(session_name)
                    .build_with_tables()
                    .unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
            Self::DocChat => {
                let session = DocumentRAGSession::new_with_session_name(session_name);
                let session_ctx = session
                    .build()
                    .with_metrics(metrics.clone())
                    .with_name(session_name)
                    .build_with_tables()
                    .unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
            Self::ToolChat => {
                let session = ToolAgentSession::new_with_session_name(session_name);
                let session_ctx = session
                    .build()
                    .with_metrics(metrics.clone())
                    .with_name(session_name)
                    .build_with_tables()
                    .unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
        }
    }

    /// Get the session stream state by name
    pub fn get_session_stream_state_by_name(
        session_plan_name: &str,
        session_name: &str,
    ) -> Result<Arc<RwLock<SessionStreamState>>> {
        if session_plan_name == Self::Chat.to_string() {
            Ok(Self::Chat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::DocChat.to_string() {
            Ok(Self::DocChat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::ToolChat.to_string() {
            Ok(Self::ToolChat.get_session_stream_state(session_name))
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }

    /// Get the session stream by name
    pub fn get_session_stream_by_name(
        session_plan_name: &str,
        session_name: &str,
        session_stream_state: Arc<RwLock<SessionStreamState>>,
        message: Option<Vec<MessageInterface>>, 
        attachment: Option<Vec<AttachmentInterface>>
    ) -> Result<SessionStream> {
        if session_plan_name == Self::Chat.to_string() {
            let incoming_message_map = create_incoming_message_map(vec![
                AvailableinterfaceSubjects::UserMessages.to_incoming_message(message, attachment, session_name)?,
            ]);
            let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
            Ok(session_stream)
        } else if session_plan_name == Self::DocChat.to_string() {
            let incoming_message_map = create_incoming_message_map(vec![
                AvailableinterfaceSubjects::UserMessages.to_incoming_message(message.clone(), None, session_name)?,
                AvailableinterfaceSubjects::UserQueries.to_incoming_message(message, None, session_name)?,
                AvailableinterfaceSubjects::UserPdf.to_incoming_message(None, attachment, session_name)?,
            ]);
            let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
            Ok(session_stream)
        } else if session_plan_name == Self::ToolChat.to_string() {
            let incoming_message_map = create_incoming_message_map(vec![
                AvailableinterfaceSubjects::UserMessages.to_incoming_message(message, None, session_name)?,
                AvailableinterfaceSubjects::UserCsv.to_incoming_message(None, attachment, session_name)?,
            ]);
            let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
            Ok(session_stream)
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }
}
