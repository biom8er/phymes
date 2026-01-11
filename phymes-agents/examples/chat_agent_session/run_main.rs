// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_diagnostics::HashMap;
use std::sync::Arc;

use phymes_agents::{
    AvailableInterfaceSubjects, ChatAgentSession, CustomAgentsBuilderTrait,
    SessionContextBuilderAgentsTrait, SessionStream, create_message_map,
};
use phymes_core::{
    AvailableSubjectsTrait, BuildableTrait, BuilderTrait, ChatBuilderTraitExt, IPCMessage,
    MappableTrait, MessageBuilderTrait, MessageTrait, TableBuilder, TableBuilderTrait,
    TablePublication, TableTrait,
};

pub async fn run_main() -> Result<()> {
    // initialize the session
    let chat_agent_session = ChatAgentSession {
        chat_api_url: Some("http://0.0.0.0:8000/v1"),
        ..Default::default()
    };
    let session_ctx = chat_agent_session
        .build()
        .with_name(chat_agent_session.session_context_name)
        .add_session_interface(None)?
        .build_with_tables()?;
    let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

    // ----- Query #1 -----
    let chat = AvailableInterfaceSubjects::UserMessages
        .to_table_builder(None)
        .append_new_user_query_str("Write a function to count prime numbers up to N.", "user")?
        .build()?;
    let message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&TablePublication::Extend {
            table_name: chat.get_name().to_string(),
        })
        .with_publisher(chat_agent_session.session_context_name)
        .make_name()?
        .build()?;
    let incoming_message_map = create_message_map(vec![message]);
    let session_stream =
        SessionStream::new(incoming_message_map, Arc::clone(&session_ctx_arc));
    let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            map.remove(&format!(
                "from_{}_on_{}",
                chat_agent_session.session_context_name,
                AvailableInterfaceSubjects::AssistantMessages
            ))
            .map(|v| v.get_message_own())
        })
        .flatten()
        .collect::<Vec<_>>();
    let json_data = TableBuilder::new_from_ipc_stream(&bytes)?
        .with_name("")
        .build()?
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    // ----- Query #2 -----
    let chat = AvailableInterfaceSubjects::UserMessages
        .to_table_builder(None)
        .append_new_user_query_str("Please provide an example using the functions.", "user")?
        .build()?;
    let message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&TablePublication::Extend {
            table_name: chat.get_name().to_string(),
        })
        .with_publisher(chat_agent_session.session_context_name)
        .make_name()?
        .build()?;
    let incoming_message_map = create_message_map(vec![message]);
    let session_stream =
        SessionStream::new(incoming_message_map, Arc::clone(&session_ctx_arc));
    let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            map.remove(&format!(
                "from_{}_on_{}",
                chat_agent_session.session_context_name,
                AvailableInterfaceSubjects::AssistantMessages
            ))
            .map(|v| v.get_message_own())
        })
        .flatten()
        .collect::<Vec<_>>();
    let json_data = TableBuilder::new_from_ipc_stream(&bytes)?
        .with_name("")
        .build()?
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    Ok(())
}
