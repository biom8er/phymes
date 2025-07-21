use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use parking_lot::RwLock;
use phymes_core::{metrics::ArrowTaskMetricsSet, session::session_context::{SessionStream, SessionStreamState}};
use serde::{Deserialize, Serialize};

use crate::candle_ops::ops_which::WhichCandleOps;

use super::{
    agent_session_builder::AgentSessionBuilderTrait, chat_agent_session::ChatAgentSession,
    document_rag_session::DocumentRAGSession, tool_agent_session::ToolAgentSession,
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

        match self {
            Self::Chat => {
                // Initialize the session
                let chat_agent_session = ChatAgentSession {
                    session_context_name: session_name,
                    chat_processor_name: "chat_processor_1",
                    chat_task_name: "chat_task_1",
                    runtime_env_name: "rt_1",
                    chat_subscription_name: "messages",
                    chat_api_url: None,
                };
                let session_ctx = chat_agent_session.make_session_context(metrics).unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
            Self::DocChat => {
                // initialize the session
                let doc_rag_session = DocumentRAGSession::new_with_session_name(session_name);
                let session_ctx = doc_rag_session
                    .make_session_context(metrics.clone())
                    .unwrap();
                Arc::new(RwLock::new(SessionStreamState::new(session_ctx)))
            }
            Self::ToolChat => {
                // initialize the session
                let tool_agent_session = ToolAgentSession {
                    session_context_name: session_name,
                    chat_processor_name: "chat_processor_1",
                    chat_task_name: "chat_task_1",
                    chat_runtime_env_name: "chat_rt_1",
                    tool_task_name: WhichCandleOps::SortScoresAndIndices.get_name(),
                    tool_processor_name: WhichCandleOps::SortScoresAndIndices.get_name(),
                    tool_runtime_env_name: "tool_rt_1",
                    summary_processor_name: "summary_processor_1",
                    hitl_task_name: WhichCandleOps::HumanInTheLoops.get_name(),
                    hitl_processor_name: WhichCandleOps::HumanInTheLoops.get_name(),
                    message_parser_task_name: "message_parser_task_1",
                    message_parser_processor_name: "message_parser_processor_1",
                    message_aggregator_task_name: "message_aggregator_task_1",
                    message_aggregator_processor_name: "message_aggregator_processor_1",
                    message_runtime_env_name: "message_rt_1",
                    state_messages_table_name: "messages",
                    state_scores_table_name: "available_data_1",
                    state_tools_table_name: "tools",
                    chat_api_url: None,
                };
                let session_ctx = tool_agent_session
                    .make_session_context(metrics.clone())
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
        user_query: &str
    ) -> SessionStream {
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        if session_plan_name == Self::Chat.get_session_plan_name() {
            SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state))
        } else if session_plan_name == Self::DocChat.get_session_plan_name() {
            SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state))
        } else if session_plan_name == Self::ToolChat.get_session_plan_name() {
            SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state))
        } else {
            panic!("Plan name {session_plan_name} was not found in the available session plans.")
        }
    }
}
