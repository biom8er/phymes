// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use std::sync::Arc;

use phymes_agents::{
    session_plans::{available_agent_subjects::{create_incoming_message_map, AvailableMessageSubscribeSubjects, AvailableMessagingPublishSubjects, MessagingPublishSubjectsTrait}, chat_agent_session::ChatAgentSession},
    session_traits::agents::{CustomAgentsBuilderTrait, SessionContextBuilderAgentsTrait},
};
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::{
        common_traits::BuilderTrait, session_context::{SessionStream, SessionStreamState},
        session_context_builder::SessionContextBuilderTrait,
    },
    table::arrow_table::ArrowTableTrait,
    task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageTrait},
};

pub async fn run_main() -> Result<()> {
    // initialize the metrics
    let metrics = ArrowTaskMetricsSet::new();

    // initialize the session
    let chat_agent_session = ChatAgentSession {
        chat_api_url: Some("http://0.0.0.0:8000/v1"),
        ..Default::default()
    };
    let session_ctx = chat_agent_session
        .build()
        .with_metrics(metrics.clone())
        .with_name(chat_agent_session.session_context_name)
        .build_with_tables()?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

    // ----- Query #1 -----
    let incoming_message_map = create_incoming_message_map(vec![
        AvailableMessagingPublishSubjects::UserMessages.to_incoming_message("Write a function to count prime numbers up to N.", chat_agent_session.session_context_name)?,
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
            chat_agent_session.session_context_name,
            AvailableMessageSubscribeSubjects::AssistantMessages
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    // ----- Query #2 -----
    session_stream_state.try_write().unwrap().set_iter(0);
    let incoming_message_map = create_incoming_message_map(vec![
        AvailableMessagingPublishSubjects::UserMessages.to_incoming_message("Please provide an example using the functions.", chat_agent_session.session_context_name)?,
    ]);
    let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
    let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
        session_stream.try_collect().await?;

    // Update the chat history with the response
    let json_data = response
        .first_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            chat_agent_session.session_context_name,
            AvailableMessageSubscribeSubjects::AssistantMessages
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    println!(
        "number of rows {}",
        metrics.clone_inner().output_rows().unwrap()
    );
    println!(
        "elasped compute {}",
        metrics.clone_inner().elapsed_compute().unwrap()
    );

    Ok(())
}
