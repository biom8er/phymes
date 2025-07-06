use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use parking_lot::RwLock;
use phymes_core::{metrics::ArrowTaskMetricsSet, session::session_context::SessionStreamState};
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
                let doc_rag_session = DocumentRAGSession {
                    // Chat tasks
                    chat_task_name: "chat_task_1",
                    message_aggregator_task_name: "message_aggregator_task_1",
                    message_aggregator_processor_name: "message_aggregator_1",
                    chat_processor_name: "chat_processor_1",
                    chat_runtime_env_name: "chat_rt_1",
                    // Embed tasks
                    embed_query_task_name: "embed_query_task_1",
                    embed_documents_task_name: "embed_documents_task_1",
                    embed_query_processor_name: "embed_query_processor_1",
                    embed_documents_processor_name: "embed_documents_processor_1",
                    document_chunk_task_name: "chunk_documents_task_1",
                    document_chunk_processor_1_name: "chunk_documents_processor_1",
                    embed_documents_runtime_env_name: "embed_documents_rt_1",
                    embed_query_runtime_env_name: "embed_query_rt_1", // "embed_documents_rt_1",
                    // Vector search tasks
                    vector_search_task_name: "vs_task_1",
                    relative_similarity_processor_name: "rel_sim_processor_1",
                    sort_scores_processor_name: "sort_scores_processor_1",
                    document_chunk_processor_2_name: "chunk_documents_processor_2", //"chunk_documents_processor_1",
                    join_chunks_processor_name: "join_scores_chunks_processor_1",
                    top_k_processor_name: "top_k_processor_1",
                    vector_search_runtime_env_name: "vs_rt_1",
                    // Session and state
                    session_context_name: session_name,
                    state_messages_table_name: "messages",
                    state_documents_table_name: "documents",
                    state_doc_embed_table_name: "doc_embeddings",
                    state_queries_table_name: "queries",
                    state_q_embed_table_name: "q_embeddings",
                    state_top_k_docs_table_name: "top_k",
                    state_scores_table_name: "tmp_scores",
                    state_scores_chunks_join_table_name: "tmp_scores_chunks_join",
                    embed_length: 1536, // Hidden size for GTE Qwen2 1.5B
                    chat_api_url: None,
                    embed_api_url: None,
                };
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
                "Plan name {} was not found in the available session plans.",
                session_plan_name
            ))
        }
    }
}
