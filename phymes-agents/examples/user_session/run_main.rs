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
    session_plans::{available_interface_subjects::{create_message_map, AvailableInterfaceSubjects}, user_session::UserSession},
    session_traits::agents::{CustomAgentsBuilderTrait, SessionContextBuilderAgentsTrait},
};
use phymes_core::{
    schemas::{available_subjects::AvailableSubjectsTrait, blob::BlobBuilderTraitExt, user::create_user_inbox_batch}, session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, session_stream::SessionStream, session_stream_state::SessionStreamState,
    }, table::{table_trait::{Table, TableBuilder, TableBuilderTrait, TableTrait}, table_publish::TablePublish}, task::message::{IPCMessage, MessageBuilderTrait, MessageTrait}
};

pub async fn run_main() -> Result<()> {
    // initialize the session
    let user_agent_session = UserSession::default();
    let session_ctx = user_agent_session
        .build()
        .with_name(user_agent_session.session_context_name)
        .build_with_tables()?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

    // Make the tabular data
    let batch = create_user_inbox_batch(vec!["contact@biom8er.com".to_string()])?;
    let bytes = Table::get_builder()
        .with_record_batches(vec![batch])?
        .with_name(AvailableInterfaceSubjects::UserJson.to_string().as_str())
        .build()?
        .to_json()?;

    // Wrap into the message
    let blob = AvailableInterfaceSubjects::UserJson.to_table_builder(None)
        .with_blob(None, Some("json"), &bytes, None)?
        .build()?;
    let blob_message = IPCMessage::get_builder()
        .with_message(blob.to_ipc_stream()?)
        .with_subject(blob.get_name())
        .with_update(&TablePublish::Replace { table_name: blob.get_name().to_string() })
        .with_publisher(user_agent_session.session_context_name)
        .make_name()?
        .build()?;
    let message_map = create_message_map(vec![blob_message]);

    let session_stream = SessionStream::new(message_map, Arc::clone(&session_stream_state));
    let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

    let attachment_data = response
        .into_iter()
        .map(|mut r| r.remove(&format!(
            "from_{}_on_{}",
            user_agent_session.session_context_name,
            AvailableInterfaceSubjects::AssistantJson
        )))
        .filter_map(|m| {
            m.map(|message| TableBuilder::new_from_ipc_stream(&message.get_message_own()).unwrap()
                .with_name("")
                .build().unwrap()
                .to_json_object().unwrap())
        })
        .flatten()
        .collect::<Vec<_>>();
    for row in &attachment_data {
        let bytes = row["bytes"].as_array().unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<u8>>();
        println!("attachment {}{}: {}", row["filename"].as_str().unwrap(), row["extension"].as_str().unwrap(), String::from_utf8_lossy(bytes.as_ref()).into_owned())
    }

    Ok(())
}
