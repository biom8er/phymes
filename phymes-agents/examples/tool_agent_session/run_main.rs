// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_agents::{
    AvailableInterfaceSubjects, CustomAgentsBuilderTrait, SessionContextBuilderAgentsTrait,
    ToolAgentSession, create_message_map,
};
use phymes_core::{
    AvailableSubjectsTrait, BlobBuilderTraitExt, BuildableTrait, BuilderTrait, ChatBuilderTraitExt,
    CsvFormat, IPCMessage, MappableTrait, MessageBuilderTrait, MessageTrait, SessionStream,
    SessionStreamState, TableBuilder, TableBuilderTrait, TablePublish, TableTrait,
};
use phymes_data::test_extract_tabular_data::make_scores_table;
use phymes_diagnostics::HashMap;
use std::sync::Arc;

pub async fn run_main() -> Result<()> {
    // initialize the session
    let tool_agent_session = ToolAgentSession::default();
    let session_ctx = tool_agent_session
        .build()
        .with_name(tool_agent_session.session_context_name)
        .build_with_tables()?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

    // Make the tabular data
    let csv_format = CsvFormat::default();
    let tabular_data = make_scores_table()?;
    let bytes = tabular_data.to_csv(csv_format.delimiter, csv_format.header)?;

    // Wrap into the message
    let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
        .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `[score]`.", "user")?
        .build()?;
    let chat_message = IPCMessage::get_builder()
        .with_message(chat.to_ipc_stream()?)
        .with_subject(chat.get_name())
        .with_update(&TablePublish::Extend {
            table_name: chat.get_name().to_string(),
        })
        .with_publisher(tool_agent_session.session_context_name)
        .make_name()?
        .build()?;
    let blob = AvailableInterfaceSubjects::UserCsv
        .to_table_builder(None)
        .with_blob(None, Some(".csv"), &bytes, None)?
        .build()?;
    let blob_message = IPCMessage::get_builder()
        .with_message(blob.to_ipc_stream()?)
        .with_subject(blob.get_name())
        .with_update(&TablePublish::Extend {
            table_name: blob.get_name().to_string(),
        })
        .with_publisher(tool_agent_session.session_context_name)
        .make_name()?
        .build()?;
    let message_map = create_message_map(vec![chat_message, blob_message]);
    let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
    let mut response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

    // Update the chat history with the response
    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            if let Some(v) = map.remove(&format!(
                "from_{}_on_{}",
                tool_agent_session.session_context_name,
                AvailableInterfaceSubjects::AssistantMessages
            )) {
                Some(v.get_message_own())
            } else {
                None
            }
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

    let bytes = response
        .iter_mut()
        .filter_map(|map| {
            if let Some(v) = map.remove(&format!(
                "from_{}_on_{}",
                tool_agent_session.session_context_name,
                AvailableInterfaceSubjects::AssistantCsv
            )) {
                Some(v.get_message_own())
            } else {
                None
            }
        })
        .flatten()
        .collect::<Vec<_>>();
    let attachment_data = TableBuilder::new_from_ipc_stream(&bytes)?
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
