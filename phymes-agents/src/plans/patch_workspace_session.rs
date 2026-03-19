/// A session for patching code workspaces from tool calls
pub struct PatchWorkspaceSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl<'a> Default for PatchWorkspaceSession<'a> {
    fn default() -> Self {
        // Create the project directory
        let session_context_name = "patch_workspace_session";

        // Initialize with reasonable default names
        Self {
            session_context_name,
        }
    }
}

impl<'a> PatchWorkspaceSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	patch_workspace_r-rt@{shape: subproc, label: patch_workspace_r}
	%% ------------------------------------------------------------------------------
	%% Apply patch to workspace
    %% - We listen for updates both on the config `apply_patch_p` subject
    %%   AND a data `WorkspacePatch` subject
    %% - The `tool_call_session` is used to trigger the operator when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph apply_patch_t
		WorkspacePatch-subject-.->|FullTable|apply_patch_p-subscribe
		Workspace-subject-.->|FullTable|apply_patch_p-subscribe
		apply_patch_p-subject-.->|LastRecordBatch|apply_patch_p-subscribe
		apply_patch_p-subscribe-->apply_patch_p-processor
		apply_patch_p-processor-->apply_patch_p-publish
		apply_patch_p-publish-->|Extend|apply_patch_s-subject
	end
	patch_workspace_r-rt-->apply_patch_t
	WorkspacePatch-subject@{shape: doc, label: WorkspacePatch}
	Workspace-subject@{shape: doc, label: Workspace}
	apply_patch_p-subject@{shape: doc, label: apply_patch_p}
	apply_patch_p-processor@{shape: rect, label: ApplyPatch}
	apply_patch_p-publish@{shape: fork}
	apply_patch_p-subscribe@{shape: diamond, label: All}
	apply_patch_s-subject@{shape: doc, label: apply_patch_s}
	%% ------------------------------------------------------------------------------"#
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    WorkspacePatch["WorkspacePatch"] {
        Utf8 filename
        Utf8 diff
        Utf8 operator
    }
    Workspace["Workspace"] {
        Utf8 path
        Utf8 content
    }
    apply_patch_p["apply_patch_p"] {
        List-UInt8 bytes
    }
    apply_patch_s["apply_patch_s"] {
        Utf8 path
        Utf8 content
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        AvailableSubjects, AvailableSubjectsTrait, BuildableTrait, BuilderTrait, IPCMessage,
        MappableTrait, MessageBuilderTrait, PatchOperator, SubjectBuilder, SubjectBuilderTrait,
        Publication, SubjectTrait, WorkspacePatchSubject, create_bytes_record_batch,
        create_workspace_batch, create_workspace_patch_batch,
    };
    use phymes_data::{AvailableCandleOperators, DataConfig, DataStreamManager};
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, ToolCallSession,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_session_w_subjects() -> Result<()> {
        // Initialize the session
        let patch_workspace_session = PatchWorkspaceSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            patch_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            patch_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(patch_workspace_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Apply patch data
        {
            // Create the mock repository
            let path = [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let content = [
                r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
                r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
                "pub mod extra;",
                r#"mod todo;
pub use todo::Todo"#,
                "pub struct Todo {}",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let batch = create_workspace_batch(path, content)?;
            let table = AvailableSubjects::Workspace.to_table(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );

            // Create the mock patches
            let filename = [
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/other.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let operator = vec![
                PatchOperator::Delete.to_string(),
                PatchOperator::Update.to_string(),
                PatchOperator::Create.to_string(),
            ];
            let content = [
                "",
                "@@ pub mod extra;\n+pub mod other;\n",
                "+pub struct Other {}\n*** End Patch",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let batch = create_workspace_patch_batch(filename, content, operator)?;
            let table = AvailableSubjects::WorkspacePatch.to_table(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );

            let apply_patch_config = DataConfig {
                lhs_name: Some(AvailableSubjects::Workspace.to_string()),
                rhs_name: Some(AvailableSubjects::WorkspacePatch.to_string()),
                lhs_values: Some(vec!["content".to_string()]),
                rhs_values: Some(vec!["diff".to_string(), "operator".to_string()]),
                lhs_pk: Some("path".to_string()),
                rhs_pk: Some("filename".to_string()),
                doc_patch: Some("[\"\"]".to_string()), // DM: equivalent of serde_json::to_string(&[serde_json::to_value("")?])?;
                cpu: false,
                operator: AvailableCandleOperators::ApplyPatch,
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            };
            let apply_patch_config_json = serde_json::to_vec(&apply_patch_config)?;
            let apply_patch_config_batch =
                create_bytes_record_batch(vec![apply_patch_config_json])?;
            let apply_patch_config_table = SubjectBuilder::new()
                .with_name("apply_patch_p")
                .with_record_batches(vec![apply_patch_config_batch])?
                .build()?;
            let _ = message_map.insert(
                apply_patch_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(apply_patch_config_table.get_name())
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(apply_patch_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: apply_patch_config_table.get_name().to_string(),
                    })
                    .with_message(apply_patch_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        {
            // Debug any errors
            let subjects_reading = session_ctx_arc.read();
            let table_reading = subjects_reading
                .subjects()
                .get(AvailableSubjects::SessionErrors.to_string().as_str())
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            let table_reading = subjects_reading
                .subjects()
                .get(AvailableSubjects::SessionTraces.to_string().as_str())
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .subjects()
                .get("apply_patch_s")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("path");
            assert_eq!(
                column,
                [
                    "/home/sandbox/Cargo.toml",
                    "/home/sandbox/src/extras/mod.rs",
                    "/home/sandbox/src/extras/todo.rs",
                    "/home/sandbox/src/lib.rs",
                    "/home/sandbox/src/extras/other.rs"
                ]
            );
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(
                column,
                [
                    "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                    "pub mod other;\nmod todo;\npub use todo::Todo",
                    "pub struct Todo {}",
                    "pub mod extra;",
                    "pub struct Other {}"
                ]
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_session_wo_subjects() -> Result<()> {
        // View task session
        let tool_call_session = ToolCallSession::new("tool_call_session", &["apply_patch_p"]);
        let tool_call_session_builder = SessionContextBuilder::from_mermaid_flowchart(
            &tool_call_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(&tool_call_session.as_mermaid_erdiagram()?, false, true)?
        .with_name(tool_call_session.session_context_name);

        // Initialize the session
        let patch_workspace_session = PatchWorkspaceSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            patch_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            patch_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(patch_workspace_session.session_context_name)
        .with_diagnostics(true)
        .extend(tool_call_session_builder)?
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Apply patch data
        {
            // Create the mock repository
            let path = [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let content = [
                r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
                r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
                "pub mod extra;",
                r#"mod todo;
pub use todo::Todo"#,
                "pub struct Todo {}",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let batch = create_workspace_batch(path, content)?;
            let table = AvailableSubjects::Workspace.to_table(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );

            // Create the mock patches
            let filename = [
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/other.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let operator = vec![
                PatchOperator::Delete.to_string(),
                PatchOperator::Update.to_string(),
                PatchOperator::Create.to_string(),
            ];
            let content = [
                "",
                "@@ pub mod extra;\n+pub mod other;\n",
                "+pub struct Other {}\n*** End Patch",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let values = filename
                .into_iter()
                .zip(content)
                .zip(operator)
                .map(|((filename, diff), operator)| {
                    let patch = WorkspacePatchSubject {
                        filename,
                        diff,
                        operator,
                    };
                    serde_json::to_value(&patch).unwrap()
                })
                .collect::<Vec<_>>();
            let doc_patch = serde_json::to_string(&values)?;

            let apply_patch_config = DataConfig {
                lhs_name: Some(AvailableSubjects::Workspace.to_string()),
                rhs_name: None,
                lhs_values: Some(vec!["content".to_string()]),
                rhs_values: Some(vec!["diff".to_string(), "operator".to_string()]),
                lhs_pk: Some("path".to_string()),
                rhs_pk: Some("filename".to_string()),
                doc_patch: Some(doc_patch),
                cpu: false,
                operator: AvailableCandleOperators::ApplyPatch,
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            };
            let apply_patch_config_json = serde_json::to_vec(&apply_patch_config)?;
            let apply_patch_config_batch =
                create_bytes_record_batch(vec![apply_patch_config_json])?;
            let apply_patch_config_table = SubjectBuilder::new()
                .with_name("apply_patch_p")
                .with_record_batches(vec![apply_patch_config_batch])?
                .build()?;
            let _ = message_map.insert(
                apply_patch_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(apply_patch_config_table.get_name())
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(apply_patch_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: apply_patch_config_table.get_name().to_string(),
                    })
                    .with_message(apply_patch_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .subjects()
                .get("apply_patch_s")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("path");
            assert_eq!(
                column,
                [
                    "/home/sandbox/Cargo.toml",
                    "/home/sandbox/src/extras/mod.rs",
                    "/home/sandbox/src/extras/todo.rs",
                    "/home/sandbox/src/lib.rs",
                    "/home/sandbox/src/extras/other.rs"
                ]
            );
            let column = table_reading.get_column_as_vec_str("content");
            assert_eq!(
                column,
                [
                    "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                    "pub mod other;\nmod todo;\npub use todo::Todo",
                    "pub struct Todo {}",
                    "pub mod extra;",
                    "pub struct Other {}"
                ]
            );
        }
        Ok(())
    }
}
