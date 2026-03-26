/// A session to sync local object storage with remote object storage
/// 
/// # Notes
/// - The syncing direction is unidirectional from remote to local where remote is
///   taken to be the ground truth
/// - Add a second `SyncContentSession` and invert local and remote names
///   to sync remote with local to achieve bidirectional syncing
pub struct SyncContentSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// Local object store metadata
    pub local_object_store_meta_name: &'a str,
    /// Remove object store metadata
    pub remote_object_store_meta_name: &'a str,
    /// Local object store
    pub local_object_store_name: &'a str,
    /// Remove object store
    pub remote_object_store_name: &'a str,
}

impl<'a> Default for SyncContentSession<'a> {
    fn default() -> Self {
        Self {
            session_context_name: "sync_content_session",
            local_object_store_meta_name: "local_object_store_meta_name",
            remote_object_store_meta_name: "remote_object_store_meta_name",
            local_object_store_name: "local_object_store_name",
            remote_object_store_name: "remote_object_store_name",
        }
    }
}

impl<'a> SyncContentSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let session_context_name = self.session_context_name;
        format!(r#"flowchart TD
	{session_context_name}_r-rt@{{shape: subproc, label: {session_context_name}_r}}
	%% ------------------------------------------------------------------------------
	%% Object store diff between local and remote
    %% 1. List remote locations
    %% 2. Diff local with remote
	%% ------------------------------------------------------------------------------
	subgraph list_remote_object_store_t
		remote_object_store_meta_s-subject-.->|AllRecordBatches|list_remote_object_store_p-subscribe
		list_remote_object_store_p-subject-.->|LastRecordBatch|list_remote_object_store_p-subscribe
		list_remote_object_store_p-subscribe-->list_remote_object_store_p-processor
		list_remote_object_store_p-processor-->list_remote_object_store_p-publish
		list_remote_object_store_p-publish-->|Replace|list_remote_object_store_s-subject
		list_remote_object_store_s-subject-->|AllRecordBatches|diff_local_remote_object_store_p-subscribe
		local_object_store_meta_s-subject-->|AllRecordBatches|diff_local_remote_object_store_p-subscribe
		diff_local_remote_object_store_p-subscribe-->diff_local_remote_object_store_p-processor
		diff_local_remote_object_store_p-processor-->diff_local_remote_object_store_p-publish
		diff_local_remote_object_store_p-publish-->|Replace|diff_local_remote_object_store_s-subject
	end
	{session_context_name}_r-rt-->list_remote_object_store_t
	remote_object_store_meta_s-subject@{{shape: doc, label: remote_object_store_meta_s}}
	list_remote_object_store_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	list_remote_object_store_p-publish@{{shape: fork}}
	list_remote_object_store_p-subscribe@{{shape: diamond, label: All}}
	list_remote_object_store_s-subject@{{shape: doc, label: list_remote_object_store_s}}
	local_object_store_meta_s-subject@{{shape: doc, label: local_object_store_meta_s}}
	diff_local_remote_object_store_p-processor@{{shape: rect, label: Diff}}
	diff_local_remote_object_store_p-publish@{{shape: fork}}
	diff_local_remote_object_store_p-subscribe@{{shape: diamond, label: All}}
	diff_local_remote_object_store_s-subject@{{shape: doc, label: diff_local_remote_object_store_s}}
	%% ------------------------------------------------------------------------------
	%% Object store remote downloads
    %% 1. Filter diff for updates and creates
    %% 2. Read updates and creates from remote
    %% 3. Write updates and creates to local
	%% ------------------------------------------------------------------------------
	subgraph get_remote_object_store_t
		diff_local_remote_object_store_s-subject-.->|AllRecordBatches|filter_create_update_remote_object_store_p-subscribe
		filter_create_update_remote_object_store_p-subscribe-->filter_create_update_remote_object_store_p-processor
		filter_create_update_remote_object_store_p-processor-->filter_create_update_remote_object_store_p-publish
		filter_create_update_remote_object_store_p-publish-->|Replace|filter_create_update_remote_object_store_s-subject
		filter_create_update_remote_object_store_s-subject-->|AllRecordBatches|get_remote_object_store_p-subscribe
		get_remote_object_store_p-subscribe-->get_remote_object_store_p-processor
		get_remote_object_store_p-processor-->get_remote_object_store_p-publish
		get_remote_object_store_p-publish-->|Replace|get_remote_object_store_s-subject        
		get_remote_object_store_s-subject-->|AllRecordBatches|put_local_object_store_p-subscribe
		put_local_object_store_p-subscribe-->put_local_object_store_p-processor
		put_local_object_store_p-processor-->put_local_object_store_p-publish
		put_local_object_store_p-publish-->|Replace|put_local_object_store_s-subject
	end
	{session_context_name}_r-rt-->get_remote_object_store_t
	filter_create_update_remote_object_store_p-processor@{{shape: rect, label: Filter}}
	filter_create_update_remote_object_store_p-publish@{{shape: fork}}
	filter_create_update_remote_object_store_p-subscribe@{{shape: diamond, label: All}}
	filter_create_update_remote_object_store_s-subject@{{shape: doc, label: filter_create_update_remote_object_store_s}}
	get_remote_object_store_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	get_remote_object_store_p-publish@{{shape: fork}}
	get_remote_object_store_p-subscribe@{{shape: diamond, label: All}}
	get_remote_object_store_s-subject@{{shape: doc, label: get_remote_object_store_s}}
	put_local_object_store_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	put_local_object_store_p-publish@{{shape: fork}}
	put_local_object_store_p-subscribe@{{shape: diamond, label: All}}
	put_local_object_store_s-subject@{{shape: doc, label: put_local_object_store_s}}
	%% ------------------------------------------------------------------------------
	%% Object store local deletes
    %% 1. Filter for deletes
    %% 2. Delete from local
	%% ------------------------------------------------------------------------------
	subgraph delete_local_object_store_t
		diff_local_remote_object_store_s-subject-.->|AllRecordBatches|filter_delete_local_object_store_p-subscribe
		filter_delete_local_object_store_p-subscribe-->filter_delete_local_object_store_p-processor
		filter_delete_local_object_store_p-processor-->filter_delete_local_object_store_p-publish
		filter_delete_local_object_store_p-publish-->|Replace|filter_delete_local_object_store_s-subject
		filter_delete_local_object_store_s-subject-->|AllRecordBatches|delete_local_object_store_p-subscribe
		delete_local_object_store_p-subscribe-->delete_local_object_store_p-processor
		delete_local_object_store_p-processor-->delete_local_object_store_p-publish
		delete_local_object_store_p-publish-->|Replace|delete_local_object_store_s-subject
	end
	{session_context_name}_r-rt-->delete_local_object_store_t
	filter_delete_local_object_store_p-processor@{{shape: rect, label: Filter}}
	filter_delete_local_object_store_p-publish@{{shape: fork}}
	filter_delete_local_object_store_p-subscribe@{{shape: diamond, label: All}}
	filter_delete_local_object_store_s-subject@{{shape: doc, label: filter_delete_local_object_store_s}}
	delete_local_object_store_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	delete_local_object_store_p-publish@{{shape: fork}}
	delete_local_object_store_p-subscribe@{{shape: diamond, label: All}}
	delete_local_object_store_s-subject@{{shape: doc, label: delete_local_object_store_s}}
	%% ------------------------------------------------------------------------------
	%% Object store local meta update
    %% 1. Patch local metadata
	%% ------------------------------------------------------------------------------
	subgraph patch_local_object_store_t
		local_object_store_meta_s-subject-.->|AllRecordBatches|patch_local_object_store_p-subscribe
		diff_local_remote_object_store_s-subject-->|AllRecordBatches|patch_local_object_store_p-subscribe
		patch_local_object_store_p-subscribe-->patch_local_object_store_p-processor
		patch_local_object_store_p-processor-->patch_local_object_store_p-publish
		patch_local_object_store_p-publish-->|Replace|local_object_store_meta_s-subject
	end
	{session_context_name}_r-rt-->patch_local_object_store_t
	patch_local_object_store_p-processor@{{shape: rect, label: ApplyPatch}}
	patch_local_object_store_p-publish@{{shape: fork}}
	patch_local_object_store_p-subscribe@{{shape: diamond, label: All}}
	%% ------------------------------------------------------------------------------
    %% Next steps
	%% - Create a new `SyncContentSession` with inverted local/remote names
    %%   for bidirectional syncing
	%% ------------------------------------------------------------------------------"#)
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        format!(r#"erDiagram
    remote_object_store_meta_s["remote_object_store_meta_s"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    list_remote_object_store_p["list_remote_object_store_p"] {{
        List-UInt8 bytes
    }}
    list_remote_object_store_s["list_remote_object_store_s"] {{
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }}"#)
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
            &sync_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &sync_content_session.as_mermaid_erdiagram(),
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
            let name = "list_remote_object_store_p";
            let messages = "remote_object_store_meta_s";
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
                query,
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
                subject_name: AvailableInterfaceSubjects::list_remote_object_store_s.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::list_remote_object_store_s.to_string().as_str())
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
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
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
            ToolCallSession::new("tool_call_session", &["list_remote_object_store_p", "download_json_p"]);
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
            &sync_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &sync_content_session.as_mermaid_erdiagram(),
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
            let name = "list_remote_object_store_p";
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
                query,
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
                subject_name: AvailableInterfaceSubjects::list_remote_object_store_s.to_string(),
            }
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::list_remote_object_store_s.to_string().as_str())
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
            .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
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
