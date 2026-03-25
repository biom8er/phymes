/// A session for syncing local object storeage with remote object storage
pub struct SyncContentSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl<'a> Default for SyncContentSession<'a> {
    fn default() -> Self {
        Self {
            session_context_name: "sync_content_session",
        }
    }
}

impl<'a> SyncContentSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	sync_content_r-rt@{shape: subproc, label: sync_content_r}
	%% ------------------------------------------------------------------------------
	%% Object store reading
    %% - We listen for updates to the remote object store reading metadata
	%% ------------------------------------------------------------------------------
	subgraph download_pdf_t
		http_client_request_pdf_s-subject-.->|AllRecordBatches|download_pdf_p-subscribe
		download_pdf_p-subject-.->|LastRecordBatch|download_pdf_p-subscribe
		download_pdf_p-subscribe-->download_pdf_p-processor
		download_pdf_p-processor-->download_pdf_p-publish
		download_pdf_p-publish-->|Extend|UserPdf-subject
	end
	sync_content_r-rt-->download_pdf_t
	http_client_request_pdf_s-subject@{shape: doc, label: http_client_request_pdf_s}
	download_pdf_p-subject@{shape: doc, label: download_pdf_p}
	download_pdf_p-processor@{shape: rect, label: HTTPClientRequestProcessor}
	download_pdf_p-publish@{shape: fork}
	download_pdf_p-subscribe@{shape: diamond, label: All}
	UserPdf-subject@{shape: doc, label: UserPdf}
	%% ------------------------------------------------------------------------------
	%% Object store writing
    %% - We listen for updates both on the config `download_json_p` subject
    %%   AND a data `http_client_request_json_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the JSON from
    %% - The `tool_call_session` is used to trigger the download when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph download_json_t
		http_client_request_json_s-subject-.->|AllRecordBatches|download_json_p-subscribe
		download_json_p-subject-.->|LastRecordBatch|download_json_p-subscribe
		download_json_p-subscribe-->download_json_p-processor
		download_json_p-processor-->download_json_p-publish
		download_json_p-publish-->|Extend|UserJson-subject
	end
	sync_content_r-rt-->download_json_t
	http_client_request_json_s-subject@{shape: doc, label: http_client_request_json_s}
	download_json_p-subject@{shape: doc, label: download_json_p}
	download_json_p-processor@{shape: rect, label: HTTPClientRequestProcessor}
	download_json_p-publish@{shape: fork}
	download_json_p-subscribe@{shape: diamond, label: All}
	UserJson-subject@{shape: doc, label: UserJson}
	%% ------------------------------------------------------------------------------
    %% Next steps
	%% - Other document downloads can be added as shown above...
	%% - Other tool calls can be integrated based on the above template...
	%% - A tool message needs to be generated based on the responses...
	%% ------------------------------------------------------------------------------"#
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> &str {
        r#"erDiagram
    http_client_request_pdf_s["http_client_request_pdf_s"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    download_pdf_p["download_pdf_p"] {
        List-UInt8 bytes
    }
    UserPdf["UserPdf"] {
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }
    http_client_request_json_s["http_client_request_json_s"] {
        Utf8 role
        Utf8 content
        Int64 timestamp
    }
    download_json_p["download_json_p"] {
        List-UInt8 bytes
    }
    UserJson["UserJson"] {
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }"#
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{
        BuildableTrait, BuilderTrait, ChatBuilderTraitExt, IPCMessage, MappableTrait,
        MessageBuilderTrait, Publication, Subject, SubjectBuilder, SubjectBuilderTrait,
        SubjectTrait, Subscription, create_bytes_record_batch,
    };
    use phymes_data::{HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType};
    use phymes_diagnostics::HashMap;

    use crate::{
        AvailableInterfaceSubjects, SessionContextBuilder, SessionContextBuilderAgentsTrait,
        SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream,
        SubscriptionTrait, ToolCallSession,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_sync_content_session_w_subjects() -> Result<()> {
        // Initialize the session
        let sync_content_session = SyncContentSession::default();
        let (session_ctx, session_messages) = SessionContextBuilder::from_mermaid_flowchart(
            sync_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            sync_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(sync_content_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        {
            let name = "download_pdf_p";
            let messages = "http_client_request_pdf_s";
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
            let http_client_config_table = SubjectBuilder::new()
                .with_name(name)
                .with_record_batches(vec![http_client_config_batch])?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(sync_content_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
            let message_builder = SubjectBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&download_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(sync_content_session.session_context_name)
                    .with_subject(messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
                    .build()?,
            );
        }

        // JSON download data
        {
            let name = "download_json_p";
            let messages = "http_client_request_json_s";
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
            let http_client_config_table = SubjectBuilder::new()
                .with_name(name)
                .with_record_batches(vec![http_client_config_batch])?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(sync_content_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
            let message_builder = SubjectBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&esearch_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(sync_content_session.session_context_name)
                    .with_subject(messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(column, ["2508.18700"]);
            let column = subject.get_column_as_vec_str("extension");
            assert_eq!(column, ["application/pdf"]);
            let column = subject.get_column_as_vec_str("metadata");
            assert_eq!(column, ["tool"]);
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = subject
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(column.len(), 505519);
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserJson.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserJson.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(
                column,
                [
                    "esearch.fcgi?db=pubmed&term=Diabetes%20Mellitus%5BMeSH%20Terms%5D%20AND%20%22Lancet%22%5BJournal%5D&retmode=json&retmax=5&mindate=2020&maxdate=2023"
                ]
            );
            let column = subject.get_column_as_vec_str("extension");
            assert_eq!(column, ["application/json; charset=UTF-8"]);
            let column = subject.get_column_as_vec_str("metadata");
            assert_eq!(column, ["tool"]);
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = subject
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(column.len(), 392);
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_sync_content_session_wo_subjects() -> Result<()> {
        // View task session
        let tool_call_session =
            ToolCallSession::new("tool_call_session", &["download_pdf_p", "download_json_p"]);
        let tool_call_session_builder = SessionContextBuilder::from_mermaid_flowchart(
            &tool_call_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &tool_call_session.as_mermaid_erdiagram()?,
            false,
            true,
        )?
        .with_name(tool_call_session.session_context_name);

        // Initialize the session
        let sync_content_session = SyncContentSession::default();
        let (session_ctx, session_messages) = SessionContextBuilder::from_mermaid_flowchart(
            sync_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            sync_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(sync_content_session.session_context_name)
        .with_diagnostics(true)
        .extend(tool_call_session_builder)?
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        {
            let name = "download_pdf_p";
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
            let http_client_config_table = SubjectBuilder::new()
                .with_name(name)
                .with_record_batches(vec![http_client_config_batch])?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(sync_content_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // JSON download data
        {
            let name = "download_json_p";
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
            let http_client_config_table = SubjectBuilder::new()
                .with_name(name)
                .with_record_batches(vec![http_client_config_batch])?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(sync_content_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(column, ["2508.18700"]);
            let column = subject.get_column_as_vec_str("extension");
            assert_eq!(column, ["application/pdf"]);
            let column = subject.get_column_as_vec_str("metadata");
            assert_eq!(column, ["tool"]);
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = subject
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(column.len(), 505519);
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserJson.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserJson.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(
                column,
                [
                    "esearch.fcgi?db=pubmed&term=Diabetes%20Mellitus%5BMeSH%20Terms%5D%20AND%20%22Lancet%22%5BJournal%5D&retmode=json&retmax=5&mindate=2020&maxdate=2023"
                ]
            );
            let column = subject.get_column_as_vec_str("extension");
            assert_eq!(column, ["application/json; charset=UTF-8"]);
            let column = subject.get_column_as_vec_str("metadata");
            assert_eq!(column, ["tool"]);
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = subject
                .get_column_as_vec_nested_primitive::<u8>("bytes")?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(column.len(), 392);
        }
        Ok(())
    }
}
