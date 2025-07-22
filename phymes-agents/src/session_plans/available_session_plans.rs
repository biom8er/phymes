use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use parking_lot::RwLock;
use phymes_core::{
    metrics::ArrowTaskMetricsSet,
    session::session_context::{SessionStream, SessionStreamState},
};
use serde::{Deserialize, Serialize};

use super::{
    agent_session_builder::AgentSessionBuilderTrait,
    chat_agent_session::{ChatAgentSession, test_chat_agent_session::bench_chat_agent_session_2},
    document_rag_session::{DocumentRAGSession, test_doc_rag_session::bench_doc_rag_session_query},
    tool_agent_session::{ToolAgentSession, test_tool_agent_session::bench_tool_agent_session},
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

impl AvailableSessionPlans {
    /// Get all available session plans
    pub fn get_all_session_plan_names() -> Vec<String> {
        let session_plans = ["Chat", "DocChat", "ToolChat"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get the session plan name
    pub fn get_session_plan_name(&self) -> &str {
        match self {
            Self::Chat => "Chat",
            Self::DocChat => "DocChat",
            Self::ToolChat => "ToolChat",
        }
    }

    /// Get the session stream state
    pub fn get_session_stream_state(&self, session_name: &str) -> Arc<RwLock<SessionStreamState>> {
        // Initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // Initialize the session
        match self {
            Self::Chat => {
                let session = ChatAgentSession::new_with_session_name(session_name);
                let session_ctx = session.make_session_context(metrics).unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
            Self::DocChat => {
                let session = DocumentRAGSession::new_with_session_name(session_name);
                let session_ctx = session.make_session_context(metrics.clone()).unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
            Self::ToolChat => {
                let session = ToolAgentSession::new_with_session_name(session_name);
                let session_ctx = session.make_session_context(metrics.clone()).unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
        }
    }

    /// Get the session stream state by name
    pub fn get_session_stream_state_by_name(
        session_plan_name: &str,
        session_name: &str,
    ) -> Result<Arc<RwLock<SessionStreamState>>> {
        if session_plan_name == Self::Chat.get_session_plan_name() {
            Ok(Self::Chat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::DocChat.get_session_plan_name() {
            Ok(Self::DocChat.get_session_stream_state(session_name))
        } else if session_plan_name == Self::ToolChat.get_session_plan_name() {
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
        user_query: &str,
    ) -> Result<SessionStream> {
        if session_plan_name == Self::Chat.get_session_plan_name() {
            let session = ChatAgentSession::new_with_session_name(session_name);
            Ok(bench_chat_agent_session_2(
                Arc::clone(&session_stream_state),
                &session,
                user_query,
            ))
        } else if session_plan_name == Self::DocChat.get_session_plan_name() {
            let session = DocumentRAGSession::new_with_session_name(session_name);
            Ok(bench_doc_rag_session_query(
                Arc::clone(&session_stream_state),
                &session,
                user_query,
            ))
        } else if session_plan_name == Self::ToolChat.get_session_plan_name() {
            let session = ToolAgentSession::new_with_session_name(session_name);
            Ok(bench_tool_agent_session(
                Arc::clone(&session_stream_state),
                &session,
                user_query,
            ))
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }
}
