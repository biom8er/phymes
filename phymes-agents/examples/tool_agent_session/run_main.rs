// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use std::sync::Arc;

use phymes_agents::{
    session_plans::{available_agent_subjects::{create_incoming_message_map, AvailableAttachmentsSubscribeSubjects, AvailableMessageSubscribeSubjects, AvailableMessagingPublishSubjects, MessagingPublishSubjectsTrait}, tool_agent_session::ToolAgentSession},
    session_traits::agents::{CustomAgentsBuilderTrait, SessionContextBuilderAgentsTrait},
};
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::{
        common_traits::{BuilderTrait, MappableTrait}, session_context::{SessionStream, SessionStreamState},
        session_context_builder::SessionContextBuilderTrait,
    },
    table::arrow_table::ArrowTableTrait,
    task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageTrait},
};

pub async fn run_main() -> Result<()> {
    // initialize the metrics
    let metrics = ArrowTaskMetricsSet::new();

    // initialize the session
    let tool_agent_session = ToolAgentSession::default();
    let session_ctx = tool_agent_session
        .build()
        .with_metrics(metrics.clone())
        .with_name(tool_agent_session.session_context_name)
        .build_with_tables()?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

    // Make the user query
    let user_query = "Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.";

    let incoming_message_map = create_incoming_message_map(vec![
        AvailableMessagingPublishSubjects::UserMessages.to_incoming_message(user_query, tool_agent_session.session_context_name)?,
    ]);
    let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
    let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
        session_stream.try_collect().await?;

    // Update the chat history with the response
    let json_data = response
        .last_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            tool_agent_session.session_context_name,
            AvailableMessageSubscribeSubjects::AssistantMessages.get_name()
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    let attachment_data = response
        .last_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            tool_agent_session.session_context_name,
            AvailableAttachmentsSubscribeSubjects::AssistantCsv.get_name()
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &attachment_data {
        let bytes = row["bytes"].as_array().unwrap()
            .into_iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<u8>>();
        println!("attachment {}.{}: {}", row["filename"], row["extension"], String::from_utf8_lossy(bytes.as_ref()).into_owned())
    }

    println!(
        "number of rows {}",
        metrics.clone_inner().output_rows().unwrap()
    );
    println!(
        "elasped compute {}",
        metrics.clone_inner().elapsed_compute().unwrap()
    );

    // println!("{:?}", session_stream_state
    //     .try_read()
    //     .unwrap()
    //     .get_session_context()
    //     .get_states()
    //     .get(SessionContextTableNames::MermaidJS.get_name())
    //     .unwrap()
    //     .try_read()
    //     .unwrap()
    //     .get_column_as_vec_str("mermaid_js_flowchart")
    //     .join(""));

    Ok(())
}
