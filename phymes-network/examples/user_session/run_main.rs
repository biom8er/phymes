#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use phymes_diagnostics::HashMap;
use std::sync::Arc;

use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait,
    SubjectTrait,
};
use phymes_event::Publication;
use phymes_message::{IPCMessage, MessageBuilderTrait, MessageTrait, create_message_map};
use phymes_network::{
    CustomAgentsBuilderTrait, NetworkBuilderAgentsTrait, NetworkStream, UserSession,
};
use phymes_schemas::{
    AttachmentBuilderTraitExt, AvailableInterfaceSubjects, AvailableSubjectsTrait,
    create_user_inbox_batch,
};

pub async fn run_main() -> Result<()> {
    // initialize the session
    let user_agent_session = UserSession::default();
    let (network, session_messages) = user_agent_session
        .build()
        .with_name(user_agent_session.network_name)
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
    let network_arc = Arc::new(network);

    // Make the tabular data
    let batch = create_user_inbox_batch(vec!["contact@biom8er.com".to_string()])?;
    let bytes = Subject::get_builder()
        .with_record_batches(vec![batch])?
        .with_name(AvailableInterfaceSubjects::UserJson.to_string().as_str())
        .build()?
        .to_json()?;

    // Wrap into the message
    let blob = AvailableInterfaceSubjects::UserJson
        .to_subject_builder(None)
        .with_attachment(None, Some("json"), &bytes, None)?
        .build()?;
    let blob_message = IPCMessage::get_builder()
        .with_message(blob.to_ipc_stream()?)
        .with_subject(blob.get_name())
        .with_update(&Publication::Replace {
            subject_name: blob.get_name().to_string(),
        })
        .with_publisher(user_agent_session.network_name)
        .make_name()?
        .build()?;
    let message_map = create_message_map(vec![blob_message]);
    let _ = network_arc
        .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
        .await;

    let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
    let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

    let attachment_data = response
        .into_iter()
        .map(|mut r| {
            r.remove(&format!(
                "from_{}_on_{}",
                user_agent_session.network_name,
                AvailableInterfaceSubjects::AssistantJson
            ))
        })
        .filter_map(|m| {
            m.map(|message| {
                SubjectBuilder::new_from_ipc_stream(&message.get_message_own())
                    .unwrap()
                    .with_name("")
                    .build()
                    .unwrap()
                    .to_json_object()
                    .unwrap()
            })
        })
        .flatten()
        .collect::<Vec<_>>();
    for row in &attachment_data {
        let bytes = row["bytes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<u8>>();
        println!(
            "attachment {}{}: {}",
            row["filename"].as_str().unwrap(),
            row["extension"].as_str().unwrap(),
            String::from_utf8_lossy(bytes.as_ref()).into_owned()
        )
    }

    Ok(())
}
