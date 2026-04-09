/// A session for patching code workspaces from tool calls
pub struct PatchWorkspaceSession<'a> {
    /// Session
    pub network_name: &'a str,
    /// Dynamic pipeline (e.g., tool call) or static pipeline
    pub is_dynamic: bool,
    /// Workspace subject name
    pub workspace_subject_name: &'a str,
    /// Patch subject name
    pub patch_subject_name: &'a str,
}

impl<'a> Default for PatchWorkspaceSession<'a> {
    fn default() -> Self {
        Self {
            network_name: "patch_workspace_session",
            is_dynamic: false,
            workspace_subject_name: "Workspace",
            patch_subject_name: "WorkspacePatch",
        }
    }
}

impl<'a> PatchWorkspaceSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let network_name = self.network_name;
        let workspace_subject_name = self.workspace_subject_name;
        let patch_subject_name = self.patch_subject_name;
        let apply_patch_p_subgraph = if self.is_dynamic {
            r#"
		apply_patch_p-subject-.->|LastRecordBatch|apply_patch_p-subscribe"#
        } else {
            ""
        };
        let apply_patch_p_subject = if self.is_dynamic {
            r#"
	apply_patch_p-subject@{shape: doc, label: apply_patch_p}"#
        } else {
            ""
        };
        format!(
            r#"flowchart TD
	{network_name}_r-rt@{{shape: subproc, label: patch_workspace_r}}
	%% ------------------------------------------------------------------------------
	%% Apply patch to workspace
    %% - We listen for updates both on the config `apply_patch_p` subject
    %%   AND a data `WorkspacePatch` subject
    %% - The `tool_call_session` is used to trigger the operator when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph apply_patch_t
        {workspace_subject_name}-subject-->|AllRecordBatches|apply_patch_p-subscribe
		{patch_subject_name}-subject-.->|AllRecordBatches|apply_patch_p-subscribe{apply_patch_p_subgraph}
		apply_patch_p-subscribe-->apply_patch_p-processor
		apply_patch_p-processor-->apply_patch_p-publish
		apply_patch_p-publish-->|Extend|apply_patch_s-subject
	end
	{network_name}_r-rt-->apply_patch_t
	{workspace_subject_name}-subject@{{shape: doc, label: {workspace_subject_name}}}
	{patch_subject_name}-subject@{{shape: doc, label: {patch_subject_name}}}{apply_patch_p_subject}
	apply_patch_p-processor@{{shape: rect, label: Patch}}
	apply_patch_p-publish@{{shape: fork}}
	apply_patch_p-subscribe@{{shape: diamond, label: All}}
	apply_patch_s-subject@{{shape: doc, label: apply_patch_s}}
	%% ------------------------------------------------------------------------------"#
        )
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        let workspace_subject_name = self.workspace_subject_name;
        let patch_subject_name = self.patch_subject_name;
        let apply_patch_p = if self.is_dynamic {
            r#"
        List-UInt8 bytes"#
                .to_string()
        } else {
            format!(
                r#"
        Utf8 lhs_name "{workspace_subject_name}"
        Utf8 rhs_name "{patch_subject_name}"
        List-Utf8 lhs_values "['content']"
        List-Utf8 rhs_values "['diff','operator']"
        Utf8 lhs_pk "path"
        Utf8 rhs_pk "filename"
        Utf8 doc_patch "['']"
        Boolean cpu "false"
        Utf8 operator "Patch"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate""#
            )
        };
        format!(
            r#"erDiagram
    {patch_subject_name}["{patch_subject_name}"] {{
        Utf8 filename
        Utf8 diff
        Utf8 operator
    }}
    {workspace_subject_name}["{workspace_subject_name}"] {{
        Utf8 path
        Utf8 content
    }}
    apply_patch_p["apply_patch_p"] {{{apply_patch_p}
    }}
    apply_patch_s["apply_patch_s"] {{
        Utf8 path
        Utf8 content
    }}"#
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilder, SubjectBuilderTrait,
        SubjectTrait,
    };
    use phymes_data::{AvailableOperators, DataConfig, DataStreamManager, PatchOperator};
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AvailableSubjects, AvailableSubjectsTrait, WorkspacePatchSubject,
        create_bytes_record_batch, create_workspace_batch, create_workspace_patch_batch,
    };
    use phymes_task::SubscriptionTrait;

    use crate::ToolCallSession;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_session_dynamic_w_subjects() -> Result<()> {
        // Initialize the session
        let patch_workspace_session = PatchWorkspaceSession {
            is_dynamic: true,
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &patch_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &patch_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(patch_workspace_session.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

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
            let table = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.network_name)
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
            let table = AvailableSubjects::WorkspacePatch.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.network_name)
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
                operator: AvailableOperators::Patch,
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
                    .with_publisher(patch_workspace_session.network_name)
                    .with_subject(apply_patch_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: apply_patch_config_table.get_name().to_string(),
                    })
                    .with_message(apply_patch_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "apply_patch_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("apply_patch_s")
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("path");
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
        let column = subject.get_column_as_vec_str("content");
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
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_session_dynamic_wo_subjects() -> Result<()> {
        // View task session
        let tool_call_session = ToolCallSession::new("tool_call_session", &["apply_patch_p"]);
        let tool_call_network_builder = NetworkBuilder::from_mermaid_flowchart(
            &tool_call_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &tool_call_session.as_mermaid_erdiagram()?,
            false,
            true,
        )?
        .with_name(tool_call_session.network_name);

        // Initialize the session
        let patch_workspace_session = PatchWorkspaceSession {
            is_dynamic: true,
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &patch_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &patch_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(patch_workspace_session.network_name)
        .with_diagnostics(true)
        .extend(tool_call_network_builder)?
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

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
            let table = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.network_name)
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
                operator: AvailableOperators::Patch,
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
                    .with_publisher(patch_workspace_session.network_name)
                    .with_subject(apply_patch_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: apply_patch_config_table.get_name().to_string(),
                    })
                    .with_message(apply_patch_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "apply_patch_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("apply_patch_s")
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("path");
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
        let column = subject.get_column_as_vec_str("content");
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
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_session_static() -> Result<()> {
        // Initialize the session
        let patch_workspace_session = PatchWorkspaceSession::default();
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &patch_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &patch_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(patch_workspace_session.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

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
            let table = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.network_name)
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
            let table = AvailableSubjects::WorkspacePatch.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(patch_workspace_session.network_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "apply_patch_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("apply_patch_s")
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("path");
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
        let column = subject.get_column_as_vec_str("content");
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
        Ok(())
    }
}
