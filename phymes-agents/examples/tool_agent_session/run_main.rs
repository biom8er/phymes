// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use std::sync::Arc;

use phymes_agents::session_plans::{
    agent_session_builder::AgentSessionBuilderTrait,
    tool_agent_session::{
        test_tool_agent_session::bench_tool_agent_session, ToolAgentSession
    },
};
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::session_context::SessionStreamState,
    table::arrow_table::ArrowTableTrait,
    task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageTrait},
};

pub async fn run_main() -> Result<()> {
    // initialize the metrics
    let metrics = ArrowTaskMetricsSet::new();

    // initialize the session
    let tool_agent_session = ToolAgentSession::default();
    let session_ctx = tool_agent_session.make_session_context(metrics.clone())?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

    // Make the user query
    let user_query = "Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.";

    let session_stream = bench_tool_agent_session(
        Arc::clone(&session_stream_state),
        &tool_agent_session,
        user_query,
    );
    let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
        session_stream.try_collect().await?;

    // Update the chat history with the response
    let json_data = response
        .last_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            tool_agent_session.session_context_name,
            tool_agent_session.state_assistant_messages_table_name
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    Ok(())
}
