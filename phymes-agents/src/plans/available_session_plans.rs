use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use phymes_core::{BuilderTrait, IPCMessageMap};
use serde::{Deserialize, Serialize};

use crate::{
    BuilderSession, ChatAgentSession, CustomAgentsBuilderTrait, DocumentRAGSession, SessionContext,
    SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderTrait,
    ToolAgentSession, UserSession,
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
    #[value(name = "Builder")]
    Builder,
    #[value(name = "Users")]
    Users,
}

impl Display for AvailableSessionPlans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "Chat"),
            Self::DocChat => write!(f, "DocChat"),
            Self::ToolChat => write!(f, "ToolChat"),
            Self::Builder => write!(f, "Builder"),
            Self::Users => write!(f, "Users"),
        }
    }
}

impl AvailableSessionPlans {
    /// Get all available session plans
    pub fn get_all_session_plan_names() -> Vec<String> {
        let session_plans = ["Chat", "DocChat", "ToolChat", "Builder"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get all available session plans
    pub fn get_deployable_session_plan_names() -> Vec<String> {
        let session_plans = ["Chat", "DocChat", "ToolChat"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get the session stream state
    pub fn get_session_context_builder(&self, session_name: &str) -> SessionContextBuilder {
        // Initialize the session context builder
        match self {
            Self::Chat => ChatAgentSession::new_with_session_name(session_name).build(),
            Self::DocChat => DocumentRAGSession::new_with_session_name(session_name).build(),
            Self::ToolChat => ToolAgentSession::new_with_session_name(session_name).build(),
            Self::Builder => BuilderSession::new_with_session_name(session_name).build(),
            Self::Users => UserSession::new_with_session_name(session_name).build(),
        }
    }

    /// Get the session stream state by name
    pub fn get_session_context_builder_by_name(
        session_plan_name: &str,
        session_name: &str,
    ) -> Result<SessionContextBuilder> {
        if session_plan_name == Self::Chat.to_string() {
            Ok(Self::Chat.get_session_context_builder(session_name))
        } else if session_plan_name == Self::DocChat.to_string() {
            Ok(Self::DocChat.get_session_context_builder(session_name))
        } else if session_plan_name == Self::ToolChat.to_string() {
            Ok(Self::ToolChat.get_session_context_builder(session_name))
        } else if session_plan_name == Self::Builder.to_string() {
            Ok(Self::Builder.get_session_context_builder(session_name))
        } else if session_plan_name == Self::Users.to_string() {
            Ok(Self::Users.get_session_context_builder(session_name))
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }

    /// Get the session stream state
    pub fn get_session_stream_state(&self, session_name: &str) -> (Arc<SessionContext>, Option<IPCMessageMap>) {
        // Initialize the session
        let builder = self.get_session_context_builder(session_name);
        let (session_ctx, message) = builder
            .with_name(session_name)
            .with_diagnostics(true)
            .add_session_interface(None)
            .unwrap()
            .add_next_tasks()
            .unwrap()
            .add_next_supersteps()
            .unwrap()
            .build_with_tables()
            .unwrap();
        (Arc::new(session_ctx), message)
    }

    /// Get the session stream state by name
    pub fn get_session_stream_state_by_name(
        session_plan_name: &str,
        session_name: &str,
    ) -> Result<(Arc<SessionContext>, Option<IPCMessageMap>)> {
        if session_plan_name == Self::Chat.to_string() {
            Ok(Self::Chat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::DocChat.to_string() {
            Ok(Self::DocChat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::ToolChat.to_string() {
            Ok(Self::ToolChat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::Builder.to_string() {
            Ok(Self::Builder.get_session_stream_state(session_name))
        } else if session_plan_name == Self::Users.to_string() {
            Ok(Self::Users.get_session_stream_state(session_name))
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }
}
