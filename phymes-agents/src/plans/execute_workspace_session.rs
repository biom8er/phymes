use anyhow::{Result, anyhow};
use phymes_data::CommandSandboxEnvironments;

use crate::{AvailableInterfaceSubjects, plans::tool_call_session::ToolSessionTrait};

/// A session for executing code workspaces
///
/// # Notes
///
/// - Specifying the schema for data_i and data_o subject is not needed because
///   `extend`ing with this session will skip duplicate subjects
///   that are already defined in the source session
///
/// # TODO
///
/// - Missing triggers for the different data_i/o methods:
///   1. None -> no `subject_name` but `cli_args`
///   2. StdIo -> `subject_name` and no `cli_args`
///   3. TempFile -> `subject_name` and no `cli_args`
pub struct ExecuteWorkspaceSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// The Temp directory for reading/writing workspace files
    pub workspace_dir: Option<String>,
    /// Input subject name
    pub subject_name_i: Option<String>,
    /// Output subject name
    pub subject_name_o: String,
    /// The environment to execute the workspace
    /// reasonable defaults in the [CommandSandboxConfig] will be filled in based on this
    ///
    /// [CommandSandboxConfig]: phymes_data::CommandSandboxConfig
    pub command_sandbox_environment: CommandSandboxEnvironments,
}

impl<'a> Default for ExecuteWorkspaceSession<'a> {
    fn default() -> Self {
        // Initialize with reasonable default names
        let session_context_name = "execute_workspace_session";
        let subject_name_o = AvailableInterfaceSubjects::AssistantCsv.to_string();
        Self {
            session_context_name,
            workspace_dir: Self::workspace_dir(None),
            subject_name_i: None,
            subject_name_o,
            command_sandbox_environment: CommandSandboxEnvironments::default(),
        }
    }
}

impl<'a> ToolSessionTrait<'a> for ExecuteWorkspaceSession<'a> {
    fn subject_names(&self) -> Vec<String> {
        let mut subject_names_vec = Vec::new();
        if let Some(subject_name_i) = self.subject_name_i.as_ref() {
            subject_names_vec.push(subject_name_i)
        }
        subject_names_vec.push(&self.subject_name_o);
        subject_names_vec.iter().map(|s| s.to_string()).collect()
    }
}

impl<'a> ExecuteWorkspaceSession<'a> {
    pub fn new(
        session_context_name: &'a str,
        workspace_dir: Option<&str>,
        subject_name_i: Option<&str>,
        subject_name_o: &str,
        command_sandbox_environment: &CommandSandboxEnvironments,
    ) -> Self {
        Self {
            session_context_name,
            workspace_dir: Self::workspace_dir(workspace_dir),
            subject_name_i: subject_name_i.map(|s| s.to_string()),
            subject_name_o: subject_name_o.to_string(),
            command_sandbox_environment: command_sandbox_environment.to_owned(),
        }
    }

    /// Workspace directory logic
    fn workspace_dir(workspace_dir: Option<&str>) -> Option<String> {
        let session_context_name = "execute_workspace_session";
        if cfg!(feature = "api") {
            #[cfg(feature = "api")]
            if let Some(workspace_dir) = workspace_dir {
                let _ = std::fs::remove_dir_all(&workspace_dir); // Doesn't matter if it is an error
                let _ = std::fs::create_dir(workspace_dir);
                Some(workspace_dir.to_string())
            } else {
                let project_dir = std::env::temp_dir().join(session_context_name);
                let _ = std::fs::remove_dir_all(&project_dir); // Doesn't matter if it is an error
                let _ = std::fs::create_dir(&project_dir);
                Some(project_dir.as_path().to_str().unwrap().to_string())
            }
        } else {
            None
        }
    }

    /// Currently, supports defaults ONLY for Python and Rust
    fn command_sandbox_p(&self) -> Result<String> {
        let mut lines = Vec::new();
        if let Some(workspace_dir) = self.workspace_dir.as_ref() {
            let line = format!(
                r#"Utf8 project_dir "{workspace_dir}"
        Utf8 data_o "TempFile"
        Utf8 initialization_file "install.sh""#
            );
            lines.push(line);
            if let Some(subject_name_i) = self.subject_name_i.as_ref() {
                let line = format!(
                    r#"
        Utf8 subject_name "{subject_name_i}"
        Utf8 data_i "TempFile""#
                );
                lines.push(line);
            } else {
                let line = format!(
                    r#"
        Utf8 data_i "None""#
                );
                lines.push(line);
            }
            match self.command_sandbox_environment {
                CommandSandboxEnvironments::Python => {
                    let line = format!(
                        r#"
        Utf8 environment "Python"
        Utf8 run_file "main.py"
        Utf8 runner "DockerUnsafe"
        Utf8 container_image "python:3.12-slim-trixie""#
                    );
                    lines.push(line);
                }
                CommandSandboxEnvironments::Rust => {
                    let line = format!(
                        r#"
        Utf8 environment "Rust"
        Utf8 run_file "main.rs"
        Utf8 runner "DockerUnsafe"
        Utf8 container_image "amd64/rust"
        List-Utf8 cli_args "['--release', '--']""#
                    );
                    lines.push(line);
                }
                _ => {
                    return Err(anyhow!(
                        "Command Sandbox Environment `{}` is not yet supported.",
                        self.command_sandbox_environment
                    ));
                }
            }
        }

        // Same for every configuration
        let line = format!(
            r#"
        Utf8 container_project_dir "/home/sandbox"
        UInt32 timeout "5"
        Utf8 workspace_name "apply_patch_s""#
        );
        lines.push(line);
        Ok(lines.join(""))
    }

    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let mut flowchart_vec = vec![
            r#"flowchart TD
	patch_workspace_r-rt@{shape: subproc, label: patch_workspace_r}"#
                .to_string(),
        ];
        if let Some(subject_name_i) = self.subject_name_i.as_ref() {
            let flowchart_component = format!(
                r#"
	%% ------------------------------------------------------------------------------
	%% Execute workspace
    %% - Listen for any changes to the updated workspace `apply_patch_s` subject
    %%   Or updates to the dataset we want to execute the workspace code on
	%% ------------------------------------------------------------------------------
	subgraph command_sandbox_t
		apply_patch_s-subject-.->|AllRecordBatches|command_sandbox_p-subscribe
        {}-subject-.->|AllRecordBatches|command_sandbox_p-subscribe
		command_sandbox_p-subscribe-->command_sandbox_p-processor
		command_sandbox_p-processor-->command_sandbox_p-publish
		command_sandbox_p-publish-->|Extend|{}-subject
	end
	patch_workspace_r-rt-->command_sandbox_t
	apply_patch_s-subject@{{shape: doc, label: apply_patch_s}}
	{}-subject@{{shape: doc, label: {}}}
	command_sandbox_p-processor@{{shape: rect, label: CommandSandboxProcessor}}
	command_sandbox_p-publish@{{shape: fork}}
	command_sandbox_p-subscribe@{{shape: diamond, label: Any}}
	{}-subject@{{shape: doc, label: {}}}
	%% ------------------------------------------------------------------------------"#,
                subject_name_i,
                self.subject_name_o,
                subject_name_i,
                subject_name_i,
                self.subject_name_o,
                self.subject_name_o
            );
            flowchart_vec.push(flowchart_component);
        } else {
            let flowchart_component = format!(
                r#"
	%% ------------------------------------------------------------------------------
	%% Execute workspace
    %% - Listen for any changes to the updated workspace `apply_patch_s` subject
	%% ------------------------------------------------------------------------------
	subgraph command_sandbox_t
		apply_patch_s-subject-.->|AllRecordBatches|command_sandbox_p-subscribe
		command_sandbox_p-subscribe-->command_sandbox_p-processor
		command_sandbox_p-processor-->command_sandbox_p-publish
		command_sandbox_p-publish-->|Extend|{}-subject
	end
	patch_workspace_r-rt-->command_sandbox_t
	apply_patch_s-subject@{{shape: doc, label: apply_patch_s}}
	command_sandbox_p-processor@{{shape: rect, label: CommandSandboxProcessor}}
	command_sandbox_p-publish@{{shape: fork}}
	command_sandbox_p-subscribe@{{shape: diamond, label: Any}}
	{}-subject@{{shape: doc, label: {}}}
	%% ------------------------------------------------------------------------------"#,
                self.subject_name_o, self.subject_name_o, self.subject_name_o
            );
            flowchart_vec.push(flowchart_component);
        }
        flowchart_vec.join("")
    }

    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> Result<String> {
        let erdiagram = format!(
            r#"erDiagram
    apply_patch_s["apply_patch_s"] {{
        Utf8 path
        Utf8 content
    }}
    {}
    command_sandbox_p["command_sandbox_p"] {{
        {}
    }}"#,
            self.erdiagram_subject_subscriptions(
                &self
                    .subject_names()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
            ),
            self.command_sandbox_p()?
        );
        Ok(erdiagram)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{
        BuildableTrait, BuilderTrait, IPCMessage, MappableTrait, MessageBuilderTrait, Publication, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait, SubjectTrait, Subscription
    };
    use phymes_data::test_command_sandbox_processor;
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream, SubscriptionTrait
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_execute_workspace_session_rs() -> Result<()> {
        // Constants
        let subject_name_i = "subject_name_i";
        let subject_name_o = "subject_name_o";
        let workspace_name = "apply_patch_s";

        // Initialize the session
        let execute_workspace_session = ExecuteWorkspaceSession::new(
            "execute_workspace_session_rs",
            None,
            Some(subject_name_i),
            subject_name_o,
            &CommandSandboxEnvironments::Rust,
        );
        let mut session_ctx_builder = SessionContextBuilder::from_mermaid_flowchart(
            &execute_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &execute_workspace_session.as_mermaid_erdiagram()?,
            false,
            true,
        )?
        .with_name(execute_workspace_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?;

        // Make the workspace data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let workspace_table = CommandSandboxEnvironments::Rust.to_default_workspace(None)?;
        let _ = message_map.insert(
            workspace_name.to_string(),
            IPCMessage::get_builder()
                .with_name(workspace_name)
                .with_publisher(execute_workspace_session.session_context_name)
                .with_subject(workspace_name)
                .with_update(&Publication::Replace {
                    subject_name: workspace_name.to_string(),
                })
                .with_message(workspace_table.to_ipc_stream()?)
                .build()?,
        );

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;
        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(subject_name_i)
            .build()?;
        let _ = message_map.insert(
            message_table.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(message_table.get_name())
                .with_publisher(execute_workspace_session.session_context_name)
                .with_subject(message_table.get_name())
                .with_update(&Publication::Replace {
                    subject_name: message_table.get_name().to_string(),
                })
                .with_message(message_table.to_ipc_stream()?)
                .build()?,
        );

        // Update place holder subjects
        let mut subjects = session_ctx_builder.subjects
            .take()
            .unwrap()
            .into_iter()
            .filter(|s| s.get_name() != subject_name_i && s.get_name() != subject_name_o)
            .collect::<Vec<_>>();
        let subject = SubjectBuilder::default()
            .with_name(message_table.get_name())
            .with_schema(message_table.get_schema())
            .with_record_batches(Vec::new())?
            .build()?;
        subjects.push(SubjectPlan::get_builder()
            .with_subject(subject)
            .build()?);
        let subject = SubjectBuilder::default()
            .with_name(subject_name_o)
            .with_schema(message_table.get_schema())
            .with_record_batches(Vec::new())?
            .build()?;
        subjects.push(SubjectPlan::get_builder()
            .with_subject(subject)
            .build()?);

        let (session_ctx, session_messages) = session_ctx_builder
            .with_subjects(subjects)
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);
        let _ = session_ctx_arc.update_subjects_from_messages(session_messages.unwrap_or_default(), 0).await;

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name:subject_name_o.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name_o)
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("name");
        assert_eq!(column, ["Alice", "Bob"]);
        let column = subject.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(column, [40, 35]);
            
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_execute_workspace_session_py() -> Result<()> {
        // Constants
        let subject_name_i = "subject_name_i";
        let subject_name_o = "subject_name_o";
        let workspace_name = "apply_patch_s";

        // Initialize the session
        let execute_workspace_session = ExecuteWorkspaceSession::new(
            "execute_workspace_session_py",
            None,
            Some(subject_name_i),
            subject_name_o,
            &CommandSandboxEnvironments::Python,
        );
        let mut session_ctx_builder = SessionContextBuilder::from_mermaid_flowchart(
            &execute_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &execute_workspace_session.as_mermaid_erdiagram()?,
            false,
            true,
        )?
        .with_name(execute_workspace_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?;

        // Make the workspace data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let workspace_table = CommandSandboxEnvironments::Python.to_default_workspace(None)?;
        let _ = message_map.insert(
            workspace_name.to_string(),
            IPCMessage::get_builder()
                .with_name(workspace_name)
                .with_publisher(execute_workspace_session.session_context_name)
                .with_subject(workspace_name)
                .with_update(&Publication::Replace {
                    subject_name: workspace_name.to_string(),
                })
                .with_message(workspace_table.to_ipc_stream()?)
                .build()?,
        );

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;
        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(subject_name_i)
            .build()?;
        let _ = message_map.insert(
            message_table.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(message_table.get_name())
                .with_publisher(execute_workspace_session.session_context_name)
                .with_subject(message_table.get_name())
                .with_update(&Publication::Replace {
                    subject_name: message_table.get_name().to_string(),
                })
                .with_message(message_table.to_ipc_stream()?)
                .build()?,
        );

        // Update place holder subjects
        let mut subjects = session_ctx_builder.subjects
            .take()
            .unwrap()
            .into_iter()
            .filter(|s| s.get_name() != subject_name_i && s.get_name() != subject_name_o)
            .collect::<Vec<_>>();
        let subject = SubjectBuilder::default()
            .with_name(message_table.get_name())
            .with_schema(message_table.get_schema())
            .with_record_batches(Vec::new())?
            .build()?;
        subjects.push(SubjectPlan::get_builder()
            .with_subject(subject)
            .build()?);
        let subject = SubjectBuilder::default()
            .with_name(subject_name_o)
            .with_schema(message_table.get_schema())
            .with_record_batches(Vec::new())?
            .build()?;
        subjects.push(SubjectPlan::get_builder()
            .with_subject(subject)
            .build()?);

        let (session_ctx, session_messages) = session_ctx_builder
            .with_subjects(subjects)
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);
        let _ = session_ctx_arc.update_subjects_from_messages(session_messages.unwrap_or_default(), 0).await;

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches { subject_name:subject_name_o.to_string() }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name_o)
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("name");
        assert_eq!(column, ["Alice", "Bob"]);
        let column = subject.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(column, [40, 35]);

        Ok(())
    }
}
