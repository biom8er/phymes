use phymes_core::ObjectStorageBackend;

/// A session to sync local object storage with remote object storage
/// 
/// # Notes
/// - The syncing direction is unidirectional from remote to local
/// - Add a second `SyncContentSession` and invert local and remote names
///   to sync remote with local to achieve bidirectional syncing
pub struct SyncContentSession<'a> {
    /// Session
    pub session_context_name: &'a str,
    /// Local object store
    pub local_object_store_name: &'a str,
    pub local_object_store_backend: &'a ObjectStorageBackend,
    pub local_object_store_bucket: Option<&'a str>,
    pub local_object_store_config: Option<&'a Map<String, Value>>,
    /// Remote object store
    pub remote_object_store_name: &'a str,
    pub remote_object_store_backend: &'a ObjectStorageBackend,
    pub remote_object_store_bucket: Option<&'a str>,
    pub remote_object_store_config: Option<&'a Map<String, Value>>,
}

impl<'a> Default for SyncContentSession<'a> {
    fn default() -> Self {
        Self {
            session_context_name: "sync_content_session",
            local_object_store_name: "local_object_store_name",
            local_object_store_backend: &ObjectStorageBackend::default(),
            local_object_store_bucket: None,
            local_object_store_config: None,
            remote_object_store_name: "remote_object_store_name",
            remote_object_store_backend: &ObjectStorageBackend::default(),
            remote_object_store_bucket: None,
            remote_object_store_config: None,
        }
    }
}

impl<'a> SyncContentSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let session_context_name = self.session_context_name;
        let local_object_store_name = self.local_object_store_name;
        let remote_object_store_name = self.remote_object_store_name;
        format!(r#"flowchart TD
	{session_context_name}_r-rt@{{shape: subproc, label: {session_context_name}_r}}
	%% ------------------------------------------------------------------------------
	%% Object store diff between local and remote
    %% 1. List remote locations
    %% 2. Diff local with remote
	%% ------------------------------------------------------------------------------
	subgraph list_{remote_object_store_name}_t
		{remote_object_store_name}_meta_s-subject-.->|AllRecordBatches|list_{remote_object_store_name}_p-subscribe
		list_{remote_object_store_name}_p-subject-.->|LastRecordBatch|list_{remote_object_store_name}_p-subscribe
		list_{remote_object_store_name}_p-subscribe-->list_{remote_object_store_name}_p-processor
		list_{remote_object_store_name}_p-processor-->list_{remote_object_store_name}_p-publish
		list_{remote_object_store_name}_p-publish-->|Replace|list_{remote_object_store_name}_s-subject
		list_{remote_object_store_name}_s-subject-->|AllRecordBatches|diff_local_{remote_object_store_name}_p-subscribe
		{local_object_store_name}_meta_s-subject-->|AllRecordBatches|diff_local_{remote_object_store_name}_p-subscribe
		diff_local_{remote_object_store_name}_p-subscribe-->diff_local_{remote_object_store_name}_p-processor
		diff_local_{remote_object_store_name}_p-processor-->diff_local_{remote_object_store_name}_p-publish
		diff_local_{remote_object_store_name}_p-publish-->|Replace|diff_local_{remote_object_store_name}_s-subject
	end
	{session_context_name}_r-rt-->list_{remote_object_store_name}_t
	{remote_object_store_name}_meta_s-subject@{{shape: doc, label: {remote_object_store_name}_meta_s}}
	list_{remote_object_store_name}_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	list_{remote_object_store_name}_p-publish@{{shape: fork}}
	list_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	list_{remote_object_store_name}_s-subject@{{shape: doc, label: list_{remote_object_store_name}_s}}
	{local_object_store_name}_meta_s-subject@{{shape: doc, label: {local_object_store_name}_meta_s}}
	diff_local_{remote_object_store_name}_p-processor@{{shape: rect, label: Diff}}
	diff_local_{remote_object_store_name}_p-publish@{{shape: fork}}
	diff_local_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	diff_local_{remote_object_store_name}_s-subject@{{shape: doc, label: diff_local_{remote_object_store_name}_s}}
	%% ------------------------------------------------------------------------------
	%% Object store remote downloads
    %% 1. Comparison columns for updates and create
    %% 2. Filter diff for updates and creates
    %% 3. Select columns
    %% 4. Read updates and creates from remote
    %% 5. Write updates and creates to local
	%% ------------------------------------------------------------------------------
	subgraph get_{remote_object_store_name}_t
		diff_local_{remote_object_store_name}_s-subject-.->|AllRecordBatches|cmp_create_update_{remote_object_store_name}_p-subscribe
		cmp_create_update_{remote_object_store_name}_p-subscribe-->cmp_create_update_{remote_object_store_name}_p-processor
		cmp_create_update_{remote_object_store_name}_p-processor-->cmp_create_update_{remote_object_store_name}_p-publish
		cmp_create_update_{remote_object_store_name}_p-publish-->|Replace|cmp_create_update_{remote_object_store_name}_s-subject
		cmp_create_update_{remote_object_store_name}_s-subject-->|AllRecordBatches|filter_create_update_{remote_object_store_name}_p-subscribe
		filter_create_update_{remote_object_store_name}_p-subscribe-->filter_create_update_{remote_object_store_name}_p-processor
		filter_create_update_{remote_object_store_name}_p-processor-->filter_create_update_{remote_object_store_name}_p-publish
		filter_create_update_{remote_object_store_name}_p-publish-->|Replace|filter_create_update_{remote_object_store_name}_s-subject        
		filter_create_update_{remote_object_store_name}_s-subject--->|AllRecordBatches|select_create_update_{remote_object_store_name}_p-subscribe
		select_create_update_{remote_object_store_name}_p-subscribe-->select_create_update_{remote_object_store_name}_p-processor
		select_create_update_{remote_object_store_name}_p-processor-->select_create_update_{remote_object_store_name}_p-publish
		select_create_update_{remote_object_store_name}_p-publish-->|Replace|select_create_update_{remote_object_store_name}_s-subject
		select_create_update_{remote_object_store_name}_s-subject-->|AllRecordBatches|get_{remote_object_store_name}_p-subscribe
		get_{remote_object_store_name}_p-subscribe-->get_{remote_object_store_name}_p-processor
		get_{remote_object_store_name}_p-processor-->get_{remote_object_store_name}_p-publish
		get_{remote_object_store_name}_p-publish-->|Replace|get_{remote_object_store_name}_s-subject        
		get_{remote_object_store_name}_s-subject-->|AllRecordBatches|put_{local_object_store_name}_p-subscribe
		put_{local_object_store_name}_p-subscribe-->put_{local_object_store_name}_p-processor
		put_{local_object_store_name}_p-processor-->put_{local_object_store_name}_p-publish
		put_{local_object_store_name}_p-publish-->|Replace|put_{local_object_store_name}_s-subject
	end
	{session_context_name}_r-rt-->get_{remote_object_store_name}_t
	cmp_create_update_{remote_object_store_name}_p-processor@{{shape: rect, label: Select}}
	cmp_create_update_{remote_object_store_name}_p-publish@{{shape: fork}}
	cmp_create_update_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	cmp_create_update_{remote_object_store_name}_s-subject@{{shape: doc, label: cmp_create_update_{remote_object_store_name}_s}}
	filter_create_update_{remote_object_store_name}_p-processor@{{shape: rect, label: Filter}}
	filter_create_update_{remote_object_store_name}_p-publish@{{shape: fork}}
	filter_create_update_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	filter_create_update_{remote_object_store_name}_s-subject@{{shape: doc, label: filter_create_update_{remote_object_store_name}_s}}
    select_create_update_{remote_object_store_name}_p-processor@{{shape: rect, label: Select}}
	select_create_update_{remote_object_store_name}_p-publish@{{shape: fork}}
	select_create_update_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	select_create_update_{remote_object_store_name}_s-subject@{{shape: doc, label: select_create_update_{remote_object_store_name}_s}}
	get_{remote_object_store_name}_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	get_{remote_object_store_name}_p-publish@{{shape: fork}}
	get_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	get_{remote_object_store_name}_s-subject@{{shape: doc, label: get_{remote_object_store_name}_s}}
	put_{local_object_store_name}_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	put_{local_object_store_name}_p-publish@{{shape: fork}}
	put_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	put_{local_object_store_name}_s-subject@{{shape: doc, label: put_{local_object_store_name}_s}}
	%% ------------------------------------------------------------------------------
	%% Object store local deletes
    %% 1. Comparison columns for delete
    %% 2. Filter for deletes
    %% 3. Select columns
    %% 4. Delete from local
	%% ------------------------------------------------------------------------------
	subgraph delete_{local_object_store_name}_t
		diff_local_{remote_object_store_name}_s-subject-.->|AllRecordBatches|cmp_delete_{local_object_store_name}_p-subscribe
		cmp_delete_{local_object_store_name}_p-subscribe-->cmp_delete_{local_object_store_name}_p-processor
		cmp_delete_{local_object_store_name}_p-processor-->cmp_delete_{local_object_store_name}_p-publish
		cmp_delete_{local_object_store_name}_p-publish-->|Replace|cmp_delete_{local_object_store_name}_s-subject
		cmp_delete_{local_object_store_name}_s-subject-->|AllRecordBatches|filter_delete_{local_object_store_name}_p-subscribe
		filter_delete_{local_object_store_name}_p-subscribe-->filter_delete_{local_object_store_name}_p-processor
		filter_delete_{local_object_store_name}_p-processor-->filter_delete_{local_object_store_name}_p-publish
		filter_delete_{local_object_store_name}_p-publish-->|Replace|filter_delete_{local_object_store_name}_s-subject
		filter_delete_{local_object_store_name}_s-subject--->|AllRecordBatches|select_delete_{local_object_store_name}_p-subscribe
		select_delete_{local_object_store_name}_p-subscribe-->select_delete_{local_object_store_name}_p-processor
		select_delete_{local_object_store_name}_p-processor-->select_delete_{local_object_store_name}_p-publish
		select_delete_{local_object_store_name}_p-publish-->|Replace|select_delete_{local_object_store_name}_s-subject
		select_delete_{local_object_store_name}_s-subject-->|AllRecordBatches|delete_{local_object_store_name}_p-subscribe
		delete_{local_object_store_name}_p-subscribe-->delete_{local_object_store_name}_p-processor
		delete_{local_object_store_name}_p-processor-->delete_{local_object_store_name}_p-publish
		delete_{local_object_store_name}_p-publish-->|Replace|delete_{local_object_store_name}_s-subject
	end
	{session_context_name}_r-rt-->delete_{local_object_store_name}_t
	cmp_delete_{local_object_store_name}_p-processor@{{shape: rect, label: Select}}
	cmp_delete_{local_object_store_name}_p-publish@{{shape: fork}}
	cmp_delete_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	cmp_delete_{local_object_store_name}_s-subject@{{shape: doc, label: cmp_delete_{local_object_store_name}_s}}
	filter_delete_{local_object_store_name}_p-processor@{{shape: rect, label: Filter}}
	filter_delete_{local_object_store_name}_p-publish@{{shape: fork}}
	filter_delete_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	filter_delete_{local_object_store_name}_s-subject@{{shape: doc, label: filter_delete_{local_object_store_name}_s}}
    select_delete_{local_object_store_name}_p-processor@{{shape: rect, label: Select}}
	select_delete_{local_object_store_name}_p-publish@{{shape: fork}}
	select_delete_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	select_delete_{local_object_store_name}_s-subject@{{shape: doc, label: select_delete_{local_object_store_name}_s}}
	delete_{local_object_store_name}_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	delete_{local_object_store_name}_p-publish@{{shape: fork}}
	delete_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	delete_{local_object_store_name}_s-subject@{{shape: doc, label: delete_{local_object_store_name}_s}}
	%% ------------------------------------------------------------------------------
	%% Object store local meta update
    %% 1. Patch local metadata
	%% ------------------------------------------------------------------------------
	subgraph patch_{local_object_store_name}_t
		{local_object_store_name}_meta_s-subject-.->|AllRecordBatches|patch_{local_object_store_name}_p-subscribe
		diff_local_{remote_object_store_name}_s-subject-->|AllRecordBatches|patch_{local_object_store_name}_p-subscribe
		patch_{local_object_store_name}_p-subscribe-->patch_{local_object_store_name}_p-processor
		patch_{local_object_store_name}_p-processor-->patch_{local_object_store_name}_p-publish
		patch_{local_object_store_name}_p-publish-->|Replace|{local_object_store_name}_meta_s-subject
	end
	{session_context_name}_r-rt-->patch_{local_object_store_name}_t
	patch_{local_object_store_name}_p-processor@{{shape: rect, label: ApplyPatch}}
	patch_{local_object_store_name}_p-publish@{{shape: fork}}
	patch_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	%% ------------------------------------------------------------------------------
    %% Next steps
	%% - Create a new `SyncContentSession` with inverted local/remote names
    %%   for bidirectional syncing
	%% ------------------------------------------------------------------------------"#)
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        let session_context_name = self.session_context_name;
        let local_object_store_name = self.local_object_store_name;
        let local_object_store_backend = self.local_object_store_backend.to_string();
        let local_object_store_bucket = self.local_object_store_bucket.unwrap_or_default();
        let local_object_store_config = if let Some(config) = self.local_object_store_config{
            let config_str = serde_json::to_string(config).unwrap();
            format!(r#"Utf8 backend_config "{config_str}""#)
        } else {
            r#"Utf8 backend_config "{}""#.to_string()
        };
        let remote_object_store_name = self.remote_object_store_name;
        let remote_object_store_backend = self.remote_object_store_backend.to_string();
        let remote_object_store_bucket = self.remote_object_store_bucket.unwrap_or_default();
        let remote_object_store_config = if let Some(config) = self.remote_object_store_config{
            let config_str = serde_json::to_string(config).unwrap();
            format!(r#"Utf8 backend_config "{config_str}""#)
        } else {
            r#"Utf8 backend_config "{}""#.to_string()
        };
        format!(r#"erDiagram
    {remote_object_store_name}_meta_s["{remote_object_store_name}_meta_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    list_{remote_object_store_name}_p["list_{remote_object_store_name}_p"] {{
        List-UInt8 bytes
    }}
    list_{remote_object_store_name}_s["list_{remote_object_store_name}_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    {local_object_store_name}_meta_s["{local_object_store_name}_meta_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    diff_local_{remote_object_store_name}_p["diff_local_{remote_object_store_name}_p"] {{
        Utf8 lhs_name "{local_object_store_name}_meta_s"
        Utf8 rhs_name "list_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['bucket', 'e_tag', 'version', 'size', 'last_modified']"
        List-Utf8 rhs_values "['bucket', 'e_tag', 'version', 'size', 'last_modified']"
        Utf8 lhs_pk "location"
        Utf8 rhs_pk "location"
        Utf8 diff "Map"
        Utf8 operator "Diff"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
    }}
    diff_local_{remote_object_store_name}_s["diff_local_{remote_object_store_name}_s"] {{
        Utf8 filename
        Utf8 diff
        Utf8 operator
    }}
    cmp_create_update_{remote_object_store_name}_p["cmp_create_update_{remote_object_store_name}_p"] {{
	    List-Utf8 as_columns "['','','','create','update']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','Create','Update']"
	    List-Utf8 column_operators "['None','None','None','Value','Value']"
	    List-Utf8 lhs_values "['filename','diff','operator','create','update']"
        Boolean cpu "false"
        Utf8 lhs_name "diff_local_{remote_object_store_name}_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    filter_create_update_{remote_object_store_name}_p["filter_create_update_{remote_object_store_name}_p"] {{
        List-Utf8 cmp_columns "['create','update']"
        List-Utf8 cmp_operators "['Like','Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_create_update_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['operator','operator']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }}
    select_create_update_{remote_object_store_name}_p["select_create_update_{remote_object_store_name}_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "filter_create_update_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['filename','diff','operator']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    get_{remote_object_store_name}_p["get_{remote_object_store_name}_p"] {{
        UInt32 timeout "15"
        Utf8 ops_type "Get"
        Utf8 backend "{remote_object_store_backend}"
        Utf8 bucket "{remote_object_store_bucket}"
        Utf8 backend_config "{remote_object_store_config}"
        Utf8 subject_name "select_create_update_{remote_object_store_name}_s"
    }}
    put_{local_object_store_name}_p["put_{local_object_store_name}_p"] {{
        UInt32 timeout "15"
        Utf8 ops_type "Put"
        Utf8 backend "{local_object_store_backend}"
        Utf8 bucket "{local_object_store_bucket}"
        Utf8 backend_config "{local_object_store_config}"
        Utf8 subject_name "get_{remote_object_store_name}_s"
    }}
    put_{local_object_store_name}_s["put_{local_object_store_name}_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    cmp_delete_{local_object_store_name}_p["cmp_delete_{local_object_store_name}_p"] {{
	    List-Utf8 as_columns "['','','','delete']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','Delete']"
	    List-Utf8 column_operators "['None','None','None','Value']"
	    List-Utf8 lhs_values "['filename','diff','operator','delete']"
        Boolean cpu "false"
        Utf8 lhs_name "diff_local_{remote_object_store_name}_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    filter_delete_{local_object_store_name}_p["filter_delete_{local_object_store_name}_p"] {{
        List-Utf8 cmp_columns "['delete']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_delete_{local_object_store_name}_s"
        List-Utf8 lhs_values "['operator','operator']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }}
    select_delete_{local_object_store_name}_p["select_delete_{local_object_store_name}_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "filter_delete_{local_object_store_name}_s"
        List-Utf8 lhs_values "['filename','diff','operator']"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    delete_{local_object_store_name}_p["delete_{local_object_store_name}_p"] {{
        UInt32 timeout "15"
        Utf8 ops_type "Delete"
        Utf8 backend "{local_object_store_backend}"
        Utf8 bucket "{local_object_store_bucket}"
        Utf8 backend_config "{local_object_store_config}"
        Utf8 subject_name "select_delete_{local_object_store_name}_s"
    }}
    delete_{local_object_store_name}_s["delete_{local_object_store_name}_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    patch_{local_object_store_name}_p["patch_{local_object_store_name}_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "{local_object_store_name}_meta_s"
        Utf8 rhs_name "diff_local_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['bucket', 'e_tag', 'version', 'size', 'last_modified']"
        List-Utf8 rhs_values "['diff', 'operator']"
        Utf8 lhs_pk "location"
        Utf8 rhs_pk "filename"
        Utf8 doc_patch ""
        Utf8 operator "ApplyPatch"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
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
