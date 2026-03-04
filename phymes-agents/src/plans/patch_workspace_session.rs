/// A session for patching and executing code workspaces
pub struct PatchWorkspaceSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// The Temp directory for reading/writing workspace files
    pub workspace_dir: Option<String>,
}

impl<'a> Default for PatchWorkspaceSession<'a> {
    fn default() -> Self {
        // Create the project directory
        let session_context_name = "patch_workspace_session";
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

        // Initialize with reasonable default names
        Self {
            session_context_name,
            workspace_dir
        }
    }
}

impl<'a> PatchWorkspaceSession<'a> {
    fn workspace_erdiagram_column(&self) -> String {
        if let Some(workspace_dir) = self.workspace_dir.as_ref() {
            format!(r#"\n\t\tUtf8 project_dir "{workspace_dir}"\n\t\t"#)
        } else {
            String::new()
        }
    }
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	patch_workspace_r-rt@{shape: subproc, label: patch_workspace_r}
	%% ------------------------------------------------------------------------------
	%% Tool call processor that enables calling processors from their config
	%% ------------------------------------------------------------------------------
	subgraph call_processor_t
        select_tasks_processors_subscriptions_publications_aggregated_s-subject-.->|FullTable|echo_processor_p-subscribe
		echo_processor_p-subscribe-->echo_processor_p-processor
		echo_processor_p-processor-->echo_processor_p-publish
		echo_processor_p-publish-->|Extend|select_tasks_processors_subscriptions_publications_aggregated_s-subject
        select_tasks_processors_subscriptions_publications_aggregated_s-subject-->|FullTable|call_processor_p-subscribe
		apply_patch_p-subject-.->|LastRecordBatch|call_processor_p-subscribe
		command_sandbox_p-subject-.->|LastRecordBatch|call_processor_p-subscribe
		call_processor_p-subscribe-->call_processor_p-processor
		call_processor_p-processor-->call_processor_p-publish
		call_processor_p-publish-->|Extend|SessionTasksSubscribePublish-subject
	end
	patch_workspace_r-rt-->call_processor_t
	select_tasks_processors_subscriptions_publications_aggregated_s-subject@{shape: doc, label: select_tasks_processors_subscriptions_publications_aggregated_s}
	echo_processor_p-processor@{shape: rect, label: ProcessorEcho}
	echo_processor_p-publish@{shape: fork}
	echo_processor_p-subscribe@{shape: diamond, label: All}
	call_processor_p-processor@{shape: rect, label: ToolCallProcessor}
	call_processor_p-publish@{shape: fork}
	call_processor_p-subscribe@{shape: diamond, label: Any}
	SessionTasksSubscribePublish-subject@{shape: doc, label: SessionTasksSubscribePublish}
	%% ------------------------------------------------------------------------------
	%% Apply patch to workspace
    %% - We listen for updates both on the config `apply_patch_p` subject
    %%   AND a data `workspace_patch_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the PDF from
    %% - The `view_task_session` is used to trigger the operator when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph apply_patch_t
		workspace_patch_s-subject-.->|FullTable|apply_patch_p-subscribe
		apply_patch_p-subject-.->|LastRecordBatch|apply_patch_p-subscribe
		apply_patch_p-subscribe-->apply_patch_p-processor
		apply_patch_p-processor-->apply_patch_p-publish
		apply_patch_p-publish-->|Extend|apply_patch_s-subject
	end
	patch_workspace_r-rt-->apply_patch_t
	workspace_patch_s-subject@{shape: doc, label: workspace_patch_s}
	apply_patch_p-subject@{shape: doc, label: apply_patch_p}
	apply_patch_p-processor@{shape: rect, label: HTTPClientRequestProcessor}
	apply_patch_p-publish@{shape: fork}
	apply_patch_p-subscribe@{shape: diamond, label: All}
	apply_patch_s-subject@{shape: doc, label: apply_patch_s}
	%% ------------------------------------------------------------------------------
	%% Execute workspace (Requires `feature = "api"`)
	%% ------------------------------------------------------------------------------
	subgraph command_sandbox_t
		apply_patch_s-subject-.->|FullTable|command_sandbox_p-subscribe
		workspace_data_s-subject-.->|FullTable|command_sandbox_p-subscribe
		command_sandbox_p-subscribe-->command_sandbox_p-processor
		command_sandbox_p-processor-->command_sandbox_p-publish
		command_sandbox_p-publish-->|Extend|command_sandbox_s-subject
	end
	patch_workspace_r-rt-->command_sandbox_t
	workspace_data_s-subject@{shape: doc, label: workspace_data_s}
	command_sandbox_p-processor@{shape: rect, label: HTTPClientRequestProcessor}
	command_sandbox_p-publish@{shape: fork}
	command_sandbox_p-subscribe@{shape: diamond, label: Any}
	command_sandbox_s-subject@{shape: doc, label: command_sandbox_s}
	%% ------------------------------------------------------------------------------
	%% Other document downloads can be added as shown above...
	%% ------------------------------------------------------------------------------"#
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        format!(r#"erDiagram
    select_tasks_processors_subscriptions_publications_aggregated_s["select_tasks_processors_subscriptions_publications_aggregated_s"] {{
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
        List-Utf8 publication_names
        List-Utf8 publication_table_names
    }}
    call_processor_p["call_processor_p"] {{
        Utf8 subject_name "select_tasks_processors_subscriptions_publications_aggregated_s"
        List-Utf8 subject_names "['apply_patch_p']"
        List-Utf8 subscription_table_names "['lhs_name', 'rhs_name', 'subject_name']"
    }}
    SessionTasksSubscribePublish["SessionTasksSubscribePublish"] {{
        Utf8 session_name
        Utf8 task_name
        Utf8 processor_name
        Utf8 processor_type
        List-Utf8 subscription_names
        List-Utf8 subscription_table_names
        List-Utf8 publication_names
        List-Utf8 publication_table_names
    }}
    workspace_patch_s["workspace_patch_s"] {{
        Utf8 path
        Utf8 content
        Utf8 operator
    }}
    apply_patch_p["apply_patch_p"] {{
        List-UInt8 bytes
    }}
    apply_patch_s["apply_patch_s"] {{
        Utf8 path
        Utf8 content
    }}
    workspace_data_s["workspace_data_s"] {{
        DM, todo: Define the schema
    }}
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
        Utf8 subject_name "workspace_data_s"
        Utf8 workspace_name "apply_patch_s"
    }}
    command_sandbox_s["command_sandbox_s"] {{
        DM, todo: update schema
    }}"#, self.workspace_erdiagram_column())
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
        ViewTaskSession,
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
            &patch_workspace_session.as_mermaid_erdiagram(),
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
                    .with_publisher(patch_workspace_session.session_context_name)
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
                    .with_publisher(patch_workspace_session.session_context_name)
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
                    .with_publisher(patch_workspace_session.session_context_name)
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
                    .with_publisher(patch_workspace_session.session_context_name)
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

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_session_wo_subjects() -> Result<()> {
        // View task session
        let view_task_session =
            ViewTaskSession::new("view_task_session", &["apply_patch_p", "command_sandbox_p"]);
        let view_task_session_builder = SessionContextBuilder::from_mermaid_flowchart(
            &view_task_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(&view_task_session.as_mermaid_erdiagram(), false, true)?
        .with_name(view_task_session.session_context_name);

        // Initialize the session
        let patch_workspace_session = PatchWorkspaceSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            patch_workspace_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            &patch_workspace_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(patch_workspace_session.session_context_name)
        .with_diagnostics(true)
        .extend(view_task_session_builder)?
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
            let id = "2508.18700";
            let download_url = format!("pdf/{id}");
            let http_client_config = HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://arxiv.org/".to_string(),
                request_schema: HTTPClientRequestSchemas::Attachments,
                json: Some(download_url),
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
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&TablePublication::Replace {
                        table_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // JSON download data
        {
            let name = "command_sandbox_p";
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
                request_schema: HTTPClientRequestSchemas::Attachments,
                json: Some(esearch_url),
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
                    .with_publisher(patch_workspace_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&TablePublication::Replace {
                        table_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
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
