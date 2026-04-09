#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use std::sync::Arc;

use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_diagnostics::HashMap;
use phymes_event::Publication;
use phymes_message::{IPCMessage, MessageBuilderTrait, MessageTrait, create_message_map};
use phymes_network::{
    ChatAgentNetwork, CustomAgentsBuilderTrait, NetworkBuilderAgentsTrait, NetworkStream,
};
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjectsTrait};
use phymes_streams::ChatBuilderTraitExt;

pub async fn run_main() -> Result<()> {
    // initialize the session
    let chat_agent_network = ChatAgentNetwork {
        chat_api_url: Some("http://0.0.0.0:8000/v1"),
        ..Default::default()
    };
    let (network, session_messages) = chat_agent_network
        .build()
        .with_name(chat_agent_network.network_name)
        .add_session_interface(None)?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
    let network_arc = Arc::new(network);

    // ----- Query #1 -----
    let chat = AvailableInterfaceSubjects::UserMessages
        .to_subject_builder(None)
        .append_new_user_query_str("Write a function to count prime numbers up to N.", "user")?
        .build()?;
    let message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&Publication::Extend {
            subject_name: chat.get_name().to_string(),
        })
        .with_publisher(chat_agent_network.network_name)
        .make_name()?
        .build()?;
    let incoming_message_map = create_message_map(vec![message]);
    let _ = network_arc
        .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
        .await;
    let network_stream = NetworkStream::new(incoming_message_map, Arc::clone(&network_arc));
    let mut response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            map.remove(&format!(
                "from_{}_on_{}",
                chat_agent_network.network_name,
                AvailableInterfaceSubjects::AssistantMessages
            ))
            .map(|v| v.get_message_own())
        })
        .flatten()
        .collect::<Vec<_>>();
    let json_data = SubjectBuilder::new_from_ipc_stream(&bytes)?
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
        .to_subject_builder(None)
        .append_new_user_query_str("Please provide an example using the functions.", "user")?
        .build()?;
    let message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&Publication::Extend {
            subject_name: chat.get_name().to_string(),
        })
        .with_publisher(chat_agent_network.network_name)
        .make_name()?
        .build()?;
    let incoming_message_map = create_message_map(vec![message]);
    let network_stream = NetworkStream::new(incoming_message_map, Arc::clone(&network_arc));
    let mut response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            map.remove(&format!(
                "from_{}_on_{}",
                chat_agent_network.network_name,
                AvailableInterfaceSubjects::AssistantMessages
            ))
            .map(|v| v.get_message_own())
        })
        .flatten()
        .collect::<Vec<_>>();
    let json_data = SubjectBuilder::new_from_ipc_stream(&bytes)?
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
