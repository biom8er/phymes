#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::test_extract_tabular_data::make_scores_table;
use phymes_diagnostics::HashMap;
use phymes_event::Publication;
use phymes_message::{IPCMessage, MessageBuilderTrait, MessageTrait, create_message_map};
use phymes_network::{
    NetworkBuilderCustomTrait, NetworkBuilderAppsTrait, NetworkStream, ToolAgentNetwork,
};
use phymes_schemas::{
    AttachmentBuilderTraitExt, AvailableInterfaceSubjects, AvailableSubjectsTrait, CsvFormat,
};
use phymes_streams::ChatBuilderTraitExt;
use std::sync::Arc;

pub async fn run_main() -> Result<()> {
    // initialize the session
    let tool_agent_network = ToolAgentNetwork::default();
    let (network, session_messages) = tool_agent_network
        .build()
        .with_name(tool_agent_network.network_name)
        .add_network_interface(None)?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
    let network_arc = Arc::new(network);

    // Make the tabular data
    let csv_format = CsvFormat::default();
    let tabular_data = make_scores_table()?;
    let bytes = tabular_data.to_csv(csv_format.delimiter, csv_format.header)?;

    // Wrap into the message
    let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
        .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `[score]`.", "user")?
        .build()?;
    let chat_message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&Publication::Extend {
            subject_name: chat.get_name().to_string(),
        })
        .with_publisher(tool_agent_network.network_name)
        .make_name()?
        .build()?;
    let blob = AvailableInterfaceSubjects::UserCsv
        .to_subject_builder(None)
        .with_attachment(None, Some("csv"), &bytes, None)?
        .build()?;
    let blob_message = IPCMessage::get_builder()
        .with_message(blob.to_ipc_stream()?)
        .with_subject(blob.get_name())
        .with_update(&Publication::Extend {
            subject_name: blob.get_name().to_string(),
        })
        .with_publisher(tool_agent_network.network_name)
        .make_name()?
        .build()?;
    let message_map = create_message_map(vec![chat_message, blob_message]);
    let _ = network_arc
        .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
        .await;
    let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
    let mut response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            map.remove(&format!(
                "from_{}_on_{}",
                tool_agent_network.network_name,
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

    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            map.remove(&format!(
                "from_{}_on_{}",
                tool_agent_network.network_name,
                AvailableInterfaceSubjects::AssistantCsv
            ))
            .map(|v| v.get_message_own())
        })
        .flatten()
        .collect::<Vec<_>>();
    let attachment_data = SubjectBuilder::new_from_ipc_stream(&bytes)?
        .with_name("")
        .build()?
        .to_json_object()?;
    for row in &attachment_data {
        let bytes = row["bytes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<u8>>();
        println!(
            "attachment {}.{}: {}",
            row["filename"],
            row["extension"],
            String::from_utf8_lossy(bytes.as_ref()).into_owned()
        )
    }

    Ok(())
}
