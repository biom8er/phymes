use phymes_data::CommandSandboxEnvironments;

use crate::{AvailableInterfaceSubjects, plans::tool_call_session::ToolSessionTrait};

/// A session for executing code workspaces
/// 
/// # Notes
/// 
/// - It is assumed that the data_i and data_o subject schemas will be overwritten
///   when this session is `extended` to the base session
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
    /// Execution environment
    pub environment: CommandSandboxEnvironments,

}

impl<'a> Default for ExecuteWorkspaceSession<'a> {
    fn default() -> Self {
        // Create the project directory
        let session_context_name = "execute_workspace_session";
        let workspace_dir = if cfg!(feature = "api") {
            #[cfg(feature = "api")]
            {
                let project_dir = std::env::temp_dir().join(session_context_name);
                let _ = std::fs::remove_dir_all(&project_dir); // Doesn't matter if it is an error
                let err = format!("Failed to create project directory at `{}`.", project_dir.as_path().to_str().unwrap());
                std::fs::create_dir(&project_dir).expect(err.as_str());
                Some(project_dir.as_path().to_str().unwrap().to_string())
            }            
        } else {
            None
        };
        let subject_name_o = AvailableInterfaceSubjects::ToolMessages.to_string();

        // Initialize with reasonable default names
        Self {
            session_context_name,
            workspace_dir,
            subject_name_i: None,
            subject_name_o
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
    pub fn new(session_context_name: &'a str, workspace_dir: Option<&str>, subject_name_i: Option<&str>, subject_name_o: &str) -> Self {
        Self {
            session_context_name,
            workspace_dir: workspace_dir.map(|dir| dir.to_string()),
            subject_name_i: subject_name_i.map(|s| s.to_string()),
            subject_name_o: subject_name_o.to_string(),
        }
    }
    fn erdiagram_workspace_columns(&self) -> String {
        if let Some(workspace_dir) = self.workspace_dir.as_ref() {
            format!(r#"
        Utf8 project_dir "{workspace_dir}"
        "#)
        } else {
            String::new()
        }
    }
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let mut flowchart_vec = vec![r#"flowchart TD
	patch_workspace_r-rt@{{shape: subproc, label: patch_workspace_r}}"#.to_string()];
        if let Some(subject_name_i) = self.subject_name_i.as_ref() {
            let flowchart_component = format!(r#"
	%% ------------------------------------------------------------------------------
	%% Execute workspace
    %% - Listen for any changes to the updated workspace `apply_patch_s` subject
    %%   Or updates to the dataset we want to execute the workspace code on
	%% ------------------------------------------------------------------------------
	subgraph command_sandbox_t
		apply_patch_s-subject-.->|FullTable|command_sandbox_p-subscribe
        {}-subject-.->|FullTable|command_sandbox_p-subscribe
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
            subject_name_i, self.subject_name_o, subject_name_i, subject_name_i, self.subject_name_o, self.subject_name_o);
            flowchart_vec.push(flowchart_component);
        } else {
            let flowchart_component = format!(r#"
	%% ------------------------------------------------------------------------------
	%% Execute workspace
    %% - Listen for any changes to the updated workspace `apply_patch_s` subject
	%% ------------------------------------------------------------------------------
	subgraph command_sandbox_t
		apply_patch_s-subject-.->|FullTable|command_sandbox_p-subscribe
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
            self.subject_name_o, self.subject_name_o, self.subject_name_o);
            flowchart_vec.push(flowchart_component);
        }
        flowchart_vec.join("")        
    }

    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        let mut erdiagram_vec = vec![r#"erDiagram
    apply_patch_s["apply_patch_s"] {{
        Utf8 path
        Utf8 content
    }}"#.to_string()];
        if let Some(subject_name_i) = self.subject_name_i.as_ref() {
            let erdiagram_component = format!(r#"
    {}
    command_sandbox_p["command_sandbox_p"] {{
        Utf8 data_i "TempFile"
        Utf8 data_o "TempFile"{}
        Utf8 container_project_dir "/home/sandbox"
        Utf8 initialization_file "install.sh"
        Utf8 run_file "main.py"
        Utf8 runner "DockerUnsafe"
        Utf8 environment "Python"
        Utf8 container_image "python:3.12-slim-trixie"
        Utf8 timeout "5"
        Utf8 subject_name "{subject_name_i}"
        Utf8 workspace_name "apply_patch_s"
    }}"#, self.erdiagram_subject_subscriptions(&self.subject_names().iter().map(|s| s.as_str()).collect::<Vec<_>>()), self.erdiagram_workspace_columns());
            erdiagram_vec.push(erdiagram_component);
        } else {
            let erdiagram_component = format!(r#"
    {}["{}"] {{
	    Utf8 role
	    Utf8 content
	    Int64 timestamp
    }}
    command_sandbox_p["command_sandbox_p"] {{
        Utf8 data_i "None"
        Utf8 data_o "None"{}
        Utf8 container_project_dir "/home/sandbox"
        Utf8 initialization_file "install.sh"
        Utf8 run_file "main.py"
        Utf8 runner "DockerUnsafe"
        Utf8 environment "Python"
        Utf8 container_image "python:3.12-slim-trixie"
        Utf8 timeout "5"
        Utf8 workspace_name "apply_patch_s"
    }}"#, self.subject_name_o, self.subject_name_o, self.erdiagram_workspace_columns());
            erdiagram_vec.push(erdiagram_component);
        }
        erdiagram_vec.join("") 
        
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        BuildableTrait, BuilderTrait, ChatBuilderTraitExt, IPCMessage, MappableTrait,
        MessageBuilderTrait, TableBuilder, TableBuilderTrait, TablePublication, TableTrait,
        create_bytes_record_batch,
    };
    use phymes_data::{HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType};
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait,
        SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_execute_workspace_session_rs() -> Result<()> {
        // Initialize the session
        let execute_workspace_session = ExecuteWorkspaceSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            &execute_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            &execute_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(execute_workspace_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        {
            let name = "apply_patch_p";
            let messages = "workspace_patch_s";
            let id = "2508.18700";
            let download_url = format!("pdf/{id}");
            let http_client_config = HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://arxiv.org/".to_string(),
                subject_name: Some(messages.to_string()),
                request_schema: HTTPClientRequestSchemas::Attachments,
                ..Default::default()
            };
            let http_client_config_json = serde_json::to_vec(&http_client_config)?;
            let http_client_config_batch =
                create_bytes_record_batch(vec![http_client_config_json])?;
            let http_client_config_table = TableBuilder::new()
                .with_name(name)
                .with_record_batches(vec![http_client_config_batch])?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(execute_workspace_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&TablePublication::Replace {
                        table_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
            let message_builder = TableBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&download_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(execute_workspace_session.session_context_name)
                    .with_subject(messages)
                    .with_update(&TablePublication::Replace {
                        table_name: messages.to_string(),
                    })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
                    .build()?,
            );
        }

        // JSON download data
        {
            let name = "command_sandbox_p";
            let messages = "workspace_data_s";
            let mesh_term = "Diabetes Mellitus";
            let year_from = 2020;
            let year_to = 2023;
            let journal_filter = Some("Lancet");
            let mut query = format!("{mesh_term}[MeSH Terms]");
            if let Some(journal) = journal_filter {
                query.push_str(&format!(" AND \"{journal}\"[Journal]"));
            }

            let esearch_url = format!(
                "db=pubmed&term={}&retmode=json&retmax=5&mindate={}&maxdate={}",
                urlencoding::encode(&query),
                year_from,
                year_to
            );
            let http_client_config = HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?".to_string(),
                subject_name: Some(messages.to_string()),
                request_schema: HTTPClientRequestSchemas::Attachments,
                ..Default::default()
            };
            let http_client_config_json = serde_json::to_vec(&http_client_config)?;
            let http_client_config_batch =
                create_bytes_record_batch(vec![http_client_config_json])?;
            let http_client_config_table = TableBuilder::new()
                .with_name(name)
                .with_record_batches(vec![http_client_config_batch])?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(execute_workspace_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&TablePublication::Replace {
                        table_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
            let message_builder = TableBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&esearch_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(execute_workspace_session.session_context_name)
                    .with_subject(messages)
                    .with_update(&TablePublication::Replace {
                        table_name: messages.to_string(),
                    })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        // {
        //     // Debug any errors
        //     let subjects_reading = session_ctx_arc.read();
        //     let table_reading = subjects_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionErrors.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        //     let table_reading = subjects_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionTraces.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        // }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get("apply_patch_s")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("filename");
            assert_eq!(column, ["2508.18700"]);
            let column = table_reading.get_column_as_vec_str("extension");
            assert_eq!(column, ["application/pdf"]);
            let column = table_reading.get_column_as_vec_str("metadata");
            assert_eq!(column, ["tool"]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = table_reading
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(column.len(), 505519);
            let table_reading = session_reading
                .get_states()
                .get("command_sandbox_s")
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("filename");
            assert_eq!(
                column,
                [
                    "esearch.fcgi?db=pubmed&term=Diabetes%20Mellitus%5BMeSH%20Terms%5D%20AND%20%22Lancet%22%5BJournal%5D&retmode=json&retmax=5&mindate=2020&maxdate=2023"
                ]
            );
            let column = table_reading.get_column_as_vec_str("extension");
            assert_eq!(column, ["application/json; charset=UTF-8"]);
            let column = table_reading.get_column_as_vec_str("metadata");
            assert_eq!(column, ["tool"]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = table_reading
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(column.len(), 392);
        }
        Ok(())
    }
}
