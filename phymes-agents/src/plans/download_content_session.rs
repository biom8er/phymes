/// A session for downloading PDF documents from a HTTP Request
pub struct DownloadContentSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl<'a> Default for DownloadContentSession<'a> {
    fn default() -> Self {
        Self {
            session_context_name: "download_content_session",
        }
    }
}

impl<'a> DownloadContentSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> &str {
        r#"flowchart TD
	%% ------------------------------------------------------------------------------
	%% PDF document downloading
    %% - We listen for updates both on the config `download_pdf_p` subject
    %%   AND a data `http_client_request_pdf_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the PDF from
    %% - The `view_task_session` is used to trigger the download when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph download_pdf_t
		http_client_request_pdf_s-subject-.->|FullTable|download_pdf_p-subscribe
		download_pdf_p-subject-.->|LastRecordBatch|download_pdf_p-subscribe
		download_pdf_p-subscribe-->download_pdf_p-processor
		download_pdf_p-processor-->download_pdf_p-publish
		download_pdf_p-publish-->|Extend|UserPdf-subject
	end
	download_content_r-rt@{shape: subproc, label: download_content_r}
	download_content_r-rt-->download_pdf_t
	http_client_request_pdf_s-subject@{shape: doc, label: http_client_request_pdf_s}
	download_pdf_p-subject@{shape: doc, label: download_pdf_p}
	download_pdf_p-processor@{shape: rect, label: HTTPClientRequestProcessor}
	download_pdf_p-publish@{shape: fork}
	download_pdf_p-subscribe@{shape: diamond, label: All}
	UserPdf-subject@{shape: doc, label: UserPdf}
	%% ------------------------------------------------------------------------------
	%% JSON document downloading
    %% - We listen for updates both on the config `download_json_p` subject
    %%   AND a data `http_client_request_json_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the JSON from
    %% - The `view_task_session` is used to trigger the download when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph download_json_t
		http_client_request_json_s-subject-.->|FullTable|download_json_p-subscribe
		download_json_p-subject-.->|LastRecordBatch|download_json_p-subscribe
		download_json_p-subscribe-->download_json_p-processor
		download_json_p-processor-->download_json_p-publish
		download_json_p-publish-->|Extend|UserJson-subject
	end
	download_content_r-rt-->download_json_t
	http_client_request_json_s-subject@{shape: doc, label: http_client_request_json_s}
	download_json_p-subject@{shape: doc, label: download_json_p}
	download_json_p-processor@{shape: rect, label: HTTPClientRequestProcessor}
	download_json_p-publish@{shape: fork}
	download_json_p-subscribe@{shape: diamond, label: All}
	UserJson-subject@{shape: doc, label: UserJson}
	%% ------------------------------------------------------------------------------
	%% Other document downloads can be added as shown above...
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
        Utf8 values
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
        Utf8 values
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
    use parking_lot::RwLock;
    use phymes_core::{
        AvailableSubjects, BuildableTrait, BuilderTrait, ChatBuilderTraitExt, IPCMessage, MappableTrait, MessageBuilderTrait, TableBuilder, TableBuilderTrait, TablePublication, TableTrait
    };
    use phymes_data::{HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType};
    use phymes_diagnostics::HashMap;

    use crate::{
        AvailableInterfaceSubjects, SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait, SessionContextBuilderTrait, SessionStream
    };

    use super::*;

    // DM, todo: test downloading from config update and from multiple messages
    #[tokio::test(flavor = "current_thread")]
    async fn test_download_content_session() -> Result<()> {
        // Initialize the session
        let download_content_session = DownloadContentSession::default();
        let session_ctx = SessionContextBuilder::from_mermaid_flowchart(
            download_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_state_from_mermaid_erdiagram(
            download_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(download_content_session.session_context_name)
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
            let name = "download_pdf_p";
            let messages = "http_client_request_pdf_s";
            let id = "2508.18700";
            let download_url = format!("pdf/{id}");
            let session_ctx_reading = session_ctx_arc.read();
            let http_client_config = HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://arxiv.org/".to_string(),
                subject_name: Some(messages.to_string()),
                request_schema: HTTPClientRequestSchemas::Blob,
                ..Default::default()
            };
            let http_client_config_json = serde_json::to_vec(&http_client_config)?;
            let http_client_config_table = TableBuilder::new()
                .with_name(name)
                .with_schema(session_ctx_reading.get_states().get(name).unwrap().read().get_schema())
                .with_json(&http_client_config_json, 1)?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(download_content_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&TablePublication::Replace { table_name: http_client_config_table.get_name().to_string() })
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
                    .with_publisher(download_content_session.session_context_name)
                    .with_subject(messages)
                    .with_update(&TablePublication::Replace { table_name: messages.to_string() })
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
            let session_ctx_reading = session_ctx_arc.read();
            let http_client_config = HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?".to_string(),
                subject_name: Some(messages.to_string()),
                request_schema: HTTPClientRequestSchemas::Messages,
                ..Default::default()
            };
            let http_client_config_json = serde_json::to_vec(&http_client_config)?;
            let http_client_config_table = TableBuilder::new()
                .with_name(name)
                .with_schema(session_ctx_reading.get_states().get(name).unwrap().read().get_schema())
                .with_json(&http_client_config_json, 1)?
                .build()?;
            let _ = message_map.insert(
                http_client_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(http_client_config_table.get_name())
                    .with_publisher(download_content_session.session_context_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&TablePublication::Replace { table_name: http_client_config_table.get_name().to_string() })
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
                    .with_publisher(download_content_session.session_context_name)
                    .with_subject(messages)
                    .with_update(&TablePublication::Replace { table_name: messages.to_string() })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
                    .build()?,
            );
        }

        // // Trigger the view task session
        // {
        //     let session_ctx_reading = session_ctx_arc.read();
        //     let table = session_ctx_reading
        //         .get_states()
        //         .get(AvailableSubjects::SessionTasks.to_string().as_str())
        //         .unwrap()
        //         .read();
        //     let session_tasks_message = IPCMessage::get_builder()
        //         .with_message(table.to_ipc_stream()?)
        //         .with_subject(AvailableSubjects::SessionTasks.to_string().as_str())
        //         .with_update(&TablePublication::Replace {
        //             table_name: AvailableSubjects::SessionTasks.to_string(),
        //         })
        //         .with_publisher(download_content_session.session_context_name)
        //         .make_name()?
        //         .build()?;
        //     let _ = message_map.insert(session_tasks_message.get_name().to_string(), session_tasks_message);
        // }

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        {
            // Debug any errors
            let subjects_reading = session_ctx_arc.read();
            let table_reading = subjects_reading
                .get_states()
                .get(AvailableSubjects::SessionErrors.to_string().as_str())
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            let table_reading = subjects_reading
                .get_states()
                .get(AvailableSubjects::SessionTraces.to_string().as_str())
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            let table_reading = subjects_reading
                .get_states()
                .get(AvailableSubjects::SubjectsChangeLog.to_string().as_str())
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            let table_reading = subjects_reading
                .get_states()
                .get("download_json_p")
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
        }

        assert_eq!(response.len(), 0);

        {
            // Test supsersteps
            let session_reading = session_ctx_arc.read();
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .unwrap()
                .read();
            println!("{}", String::from_utf8(table_reading.to_csv(b',', true)?)?);
            let column = table_reading.get_column_as_vec_str("filename");
            assert_eq!(column, [""]);
            let column = table_reading.get_column_as_vec_str("extension");
            assert_eq!(column, [""]);
            let column = table_reading.get_column_as_vec_str("metadata");
            assert_eq!(column, [""]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = table_reading.get_column_as_vec_nested_primitive::<u8>("bytes")?.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(column.len(), 0);
            let table_reading = session_reading
                .get_states()
                .get(AvailableInterfaceSubjects::UserJson.to_string().as_str())
                .unwrap()
                .read();
            let column = table_reading.get_column_as_vec_str("filename");
            assert_eq!(column, [""]);
            let column = table_reading.get_column_as_vec_str("extension");
            assert_eq!(column, [""]);
            let column = table_reading.get_column_as_vec_str("metadata");
            assert_eq!(column, [""]);
            let column = table_reading.get_column_as_vec_primitive::<i64>("timestamp")?;
            for c in column {
                assert!(c > 0);
            }
            let column = table_reading.get_column_as_vec_nested_primitive::<u8>("bytes")?.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(column.len(), 0);
        }
        Ok(())
    }
}
