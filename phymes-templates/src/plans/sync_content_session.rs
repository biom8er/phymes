use phymes_subject::ObjectStorageBackend;
use serde_json::{Map, Value};

/// A session to sync local object storage with remote object storage
///
/// # Notes
/// - The syncing direction is unidirectional from remote to local
/// - Add a second `SyncContentSession` and invert local and remote names
///   to sync remote with local to achieve bidirectional syncing
pub struct SyncContentSession<'a> {
    /// Session
    pub network_name: &'a str,
    /// Local object store
    pub local_object_store_name: &'a str,
    pub local_object_store_backend: ObjectStorageBackend,
    pub local_object_store_bucket: Option<&'a str>,
    pub local_object_store_config: Option<&'a Map<String, Value>>,
    /// Remote object store
    pub remote_object_store_name: &'a str,
    pub remote_object_store_backend: ObjectStorageBackend,
    pub remote_object_store_bucket: Option<&'a str>,
    pub remote_object_store_config: Option<&'a Map<String, Value>>,
}

impl<'a> Default for SyncContentSession<'a> {
    fn default() -> Self {
        Self {
            network_name: "sync_content_session",
            local_object_store_name: "local_object_store_name",
            local_object_store_backend: ObjectStorageBackend::default(),
            local_object_store_bucket: None,
            local_object_store_config: None,
            remote_object_store_name: "remote_object_store_name",
            remote_object_store_backend: ObjectStorageBackend::default(),
            remote_object_store_bucket: None,
            remote_object_store_config: None,
        }
    }
}

impl<'a> SyncContentSession<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let network_name = self.network_name;
        let local_object_store_name = self.local_object_store_name;
        let remote_object_store_name = self.remote_object_store_name;
        format!(
            r#"flowchart TD
	{network_name}_r-rt@{{shape: subproc, label: {network_name}_r}}
	%% ------------------------------------------------------------------------------
	%% Object store list remote locations
	%% ------------------------------------------------------------------------------
	subgraph list_{remote_object_store_name}_t
		{remote_object_store_name}_meta_s-subject-.->|AllRecordBatches|list_{remote_object_store_name}_p-subscribe
		list_{remote_object_store_name}_p-subject-.->|LastRecordBatch|list_{remote_object_store_name}_p-subscribe
		list_{remote_object_store_name}_p-subscribe-->list_{remote_object_store_name}_p-processor
		list_{remote_object_store_name}_p-processor-->list_{remote_object_store_name}_p-publish
		list_{remote_object_store_name}_p-publish-->|Replace|list_{remote_object_store_name}_s-subject
	end
	{network_name}_r-rt-->list_{remote_object_store_name}_t
	{remote_object_store_name}_meta_s-subject@{{shape: doc, label: {remote_object_store_name}_meta_s}}
	list_{remote_object_store_name}_p-subject@{{shape: doc, label: list_{remote_object_store_name}_p}}
	list_{remote_object_store_name}_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	list_{remote_object_store_name}_p-publish@{{shape: fork}}
	list_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	list_{remote_object_store_name}_s-subject@{{shape: doc, label: list_{remote_object_store_name}_s}}
	%% ------------------------------------------------------------------------------
	%% Object store diff between local and remote
	%% ------------------------------------------------------------------------------
	subgraph diff_{remote_object_store_name}_t
		list_{remote_object_store_name}_s-subject-.->|AllRecordBatches|diff_{remote_object_store_name}_p-subscribe
		{local_object_store_name}_meta_s-subject-->|AllRecordBatches|diff_{remote_object_store_name}_p-subscribe
		diff_{remote_object_store_name}_p-subscribe-->diff_{remote_object_store_name}_p-processor
		diff_{remote_object_store_name}_p-processor-->diff_{remote_object_store_name}_p-publish
		diff_{remote_object_store_name}_p-publish-->|Replace|diff_{remote_object_store_name}_s-subject
	end
	{network_name}_r-rt-->diff_{remote_object_store_name}_t
	{local_object_store_name}_meta_s-subject@{{shape: doc, label: {local_object_store_name}_meta_s}}
	diff_{remote_object_store_name}_p-processor@{{shape: rect, label: Diff}}
	diff_{remote_object_store_name}_p-publish@{{shape: fork}}
	diff_{remote_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	diff_{remote_object_store_name}_s-subject@{{shape: doc, label: diff_{remote_object_store_name}_s}}
	%% ------------------------------------------------------------------------------
	%% Object store remote downloads
    %% 1. Comparison columns for updates and create
    %% 2. Filter diff for updates and creates
    %% 3. Select columns
    %% 4. Read updates and creates from remote
    %% 5. Write updates and creates to local
	%% ------------------------------------------------------------------------------
	subgraph get_{remote_object_store_name}_t
		diff_{remote_object_store_name}_s-subject-.->|AllRecordBatches|cmp_create_update_{remote_object_store_name}_p-subscribe
		cmp_create_update_{remote_object_store_name}_p-subscribe-->cmp_create_update_{remote_object_store_name}_p-processor
		cmp_create_update_{remote_object_store_name}_p-processor-->cmp_create_update_{remote_object_store_name}_p-publish
		cmp_create_update_{remote_object_store_name}_p-publish-->|Replace|cmp_create_update_{remote_object_store_name}_s-subject
		cmp_create_update_{remote_object_store_name}_s-subject-->|AllRecordBatches|filter_create_update_{remote_object_store_name}_p-subscribe
		filter_create_update_{remote_object_store_name}_p-subscribe-->filter_create_update_{remote_object_store_name}_p-processor
		filter_create_update_{remote_object_store_name}_p-processor-->filter_create_update_{remote_object_store_name}_p-publish
		filter_create_update_{remote_object_store_name}_p-publish-->|Replace|filter_create_update_{remote_object_store_name}_s-subject        
		filter_create_update_{remote_object_store_name}_s-subject-->|AllRecordBatches|select_create_update_{remote_object_store_name}_p-subscribe
		select_create_update_{remote_object_store_name}_p-subscribe-->select_create_update_{remote_object_store_name}_p-processor
		select_create_update_{remote_object_store_name}_p-processor-->select_create_update_{remote_object_store_name}_p-publish
		select_create_update_{remote_object_store_name}_p-publish-->|Replace|select_create_update_{remote_object_store_name}_s-subject
		select_create_update_{remote_object_store_name}_s-subject-->|AllRecordBatches|get_{remote_object_store_name}_p-subscribe
		get_{remote_object_store_name}_p-subscribe-->get_{remote_object_store_name}_p-processor
		get_{remote_object_store_name}_p-processor-->get_{remote_object_store_name}_p-publish
		get_{remote_object_store_name}_p-publish-->|Replace|get_{remote_object_store_name}_s-subject
	end
	{network_name}_r-rt-->get_{remote_object_store_name}_t
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
	%% ------------------------------------------------------------------------------
	%% Object store Write updates and creates to local
    %% - Multiple calls to ObjectStoreProcessor within the same tasks is not possible
    %%   due to some issue with the Tokio runtime...
	%% ------------------------------------------------------------------------------
	subgraph put_{local_object_store_name}_t       
		get_{remote_object_store_name}_s-subject-.->|DrainRecordBatches|put_{local_object_store_name}_p-subscribe
		put_{local_object_store_name}_p-subscribe-->put_{local_object_store_name}_p-processor
		put_{local_object_store_name}_p-processor-->put_{local_object_store_name}_p-publish
		put_{local_object_store_name}_p-publish-->|Replace|put_{local_object_store_name}_s-subject
	end
	{network_name}_r-rt-->put_{local_object_store_name}_t
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
		diff_{remote_object_store_name}_s-subject-.->|AllRecordBatches|cmp_delete_{local_object_store_name}_p-subscribe
		cmp_delete_{local_object_store_name}_p-subscribe-->cmp_delete_{local_object_store_name}_p-processor
		cmp_delete_{local_object_store_name}_p-processor-->cmp_delete_{local_object_store_name}_p-publish
		cmp_delete_{local_object_store_name}_p-publish-->|Replace|cmp_delete_{local_object_store_name}_s-subject
		cmp_delete_{local_object_store_name}_s-subject-->|AllRecordBatches|filter_delete_{local_object_store_name}_p-subscribe
		filter_delete_{local_object_store_name}_p-subscribe-->filter_delete_{local_object_store_name}_p-processor
		filter_delete_{local_object_store_name}_p-processor-->filter_delete_{local_object_store_name}_p-publish
		filter_delete_{local_object_store_name}_p-publish-->|Replace|filter_delete_{local_object_store_name}_s-subject
		filter_delete_{local_object_store_name}_s-subject-->|AllRecordBatches|select_delete_{local_object_store_name}_p-subscribe
		select_delete_{local_object_store_name}_p-subscribe-->select_delete_{local_object_store_name}_p-processor
		select_delete_{local_object_store_name}_p-processor-->select_delete_{local_object_store_name}_p-publish
		select_delete_{local_object_store_name}_p-publish-->|Replace|select_delete_{local_object_store_name}_s-subject
		select_delete_{local_object_store_name}_s-subject-->|AllRecordBatches|delete_{local_object_store_name}_p-subscribe
		delete_{local_object_store_name}_p-subscribe-->delete_{local_object_store_name}_p-processor
		delete_{local_object_store_name}_p-processor-->delete_{local_object_store_name}_p-publish
		delete_{local_object_store_name}_p-publish-->|Replace|delete_{local_object_store_name}_s-subject
	end
	{network_name}_r-rt-->delete_{local_object_store_name}_t
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
		{local_object_store_name}_meta_s-subject-->|AllRecordBatches|patch_{local_object_store_name}_p-subscribe
		diff_{remote_object_store_name}_s-subject-.->|AllRecordBatches|patch_{local_object_store_name}_p-subscribe
		patch_{local_object_store_name}_p-subscribe-->patch_{local_object_store_name}_p-processor
		patch_{local_object_store_name}_p-processor-->patch_{local_object_store_name}_p-publish
		patch_{local_object_store_name}_p-publish-->|Replace|{local_object_store_name}_meta_s-subject
	end
	{network_name}_r-rt-->patch_{local_object_store_name}_t
	patch_{local_object_store_name}_p-processor@{{shape: rect, label: Patch}}
	patch_{local_object_store_name}_p-publish@{{shape: fork}}
	patch_{local_object_store_name}_p-subscribe@{{shape: diamond, label: All}}
	%% ------------------------------------------------------------------------------
    %% Next steps
	%% - Create a new `SyncContentSession` with inverted local/remote names
    %%   for bidirectional syncing
	%% ------------------------------------------------------------------------------"#
        )
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        let local_object_store_name = self.local_object_store_name;
        let local_object_store_backend = self.local_object_store_backend.to_string();
        let local_object_store_bucket = self.local_object_store_bucket.unwrap_or_default();
        let local_object_store_config = if let Some(config) = self.local_object_store_config {
            serde_json::to_string(config).unwrap().replace('"', "'")
        } else {
            "{}".to_string()
        };
        let remote_object_store_name = self.remote_object_store_name;
        let remote_object_store_backend = self.remote_object_store_backend.to_string();
        let remote_object_store_bucket = self.remote_object_store_bucket.unwrap_or_default();
        let remote_object_store_config = if let Some(config) = self.remote_object_store_config {
            serde_json::to_string(config).unwrap().replace('"', "'")
        } else {
            "{}".to_string()
        };
        format!(
            r#"erDiagram
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
    diff_{remote_object_store_name}_p["diff_{remote_object_store_name}_p"] {{
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
        Boolean cpu "false"
    }}
    diff_{remote_object_store_name}_s["diff_{remote_object_store_name}_s"] {{
        Utf8 location
        Utf8 diff
        Utf8 operator
    }}
    cmp_create_update_{remote_object_store_name}_p["cmp_create_update_{remote_object_store_name}_p"] {{
	    List-Utf8 as_columns "['','','','create','update']"
	    List-Utf8 cast_datatypes "['Utf8','Utf8','Utf8','Utf8','Utf8']"
	    List-Utf8 cast_templates "['','','','Create','Update']"
	    List-Utf8 column_operators "['None','None','None','Value','Value']"
	    List-Utf8 lhs_values "['location','diff','operator','create','update']"
        Boolean cpu "false"
        Utf8 lhs_name "diff_{remote_object_store_name}_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    filter_create_update_{remote_object_store_name}_p["filter_create_update_{remote_object_store_name}_p"] {{
        List-Utf8 cmp_columns "['create','update']"
        List-Utf8 cmp_operators "['Like','Like']"
        Utf8 cmp_predicate "Any"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_create_update_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['operator','operator']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }}
    select_create_update_{remote_object_store_name}_p["select_create_update_{remote_object_store_name}_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "filter_create_update_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['location','diff','operator']"
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
    get_{remote_object_store_name}_s["get_{remote_object_store_name}_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 metadata
        Int64 last_modified
        List-UInt8 bytes
    }}
    put_{local_object_store_name}_p["put_{local_object_store_name}_p"] {{
        UInt32 timeout "15"
        Utf8 ops_type "Put"
        Utf8 backend "{local_object_store_backend}"
        Utf8 bucket "{local_object_store_bucket}"
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
	    List-Utf8 lhs_values "['location','diff','operator','delete']"
        Boolean cpu "false"
        Utf8 lhs_name "diff_{remote_object_store_name}_s"
        Utf8 operator "Select"
        Utf8 lhs_stream "Accumulate"
    }}
    filter_delete_{local_object_store_name}_p["filter_delete_{local_object_store_name}_p"] {{
        List-Utf8 cmp_columns "['delete']"
        List-Utf8 cmp_operators "['Like']"
        Utf8 cmp_predicate "All"
        Boolean cpu "false"
        Utf8 lhs_name "cmp_delete_{local_object_store_name}_s"
        List-Utf8 lhs_values "['operator']"
        Utf8 operator "Filter"
        Utf8 lhs_stream "Accumulate"
    }}
    select_delete_{local_object_store_name}_p["select_delete_{local_object_store_name}_p"] {{
        Boolean cpu "false"
        Utf8 lhs_name "filter_delete_{local_object_store_name}_s"
        List-Utf8 lhs_values "['location','diff','operator']"
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
        Utf8 rhs_name "diff_{remote_object_store_name}_s"
        List-Utf8 lhs_values "['bucket', 'e_tag', 'version', 'size', 'last_modified']"
        List-Utf8 rhs_values "['diff', 'operator']"
        Utf8 lhs_pk "location"
        Utf8 rhs_pk "location"
        Utf8 doc_patch "['']"
        Utf8 operator "Patch"
        Utf8 lhs_stream "Accumulate"
        Utf8 rhs_stream "Accumulate"
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
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectBuilder, SubjectBuilderTrait, SubjectTrait, make_store, test_subject,
    };
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AvailableSubjects, create_bytes_record_batch, create_object_store_meta_batch,
    };
    use phymes_streams::{ObjectStoreConfig, ObjectStoreOptsType};
    use phymes_task::{PublicationTrait, SubscriptionTrait};
    #[cfg(not(target_family = "wasm"))]
    use tempfile::TempDir;

    use crate::ToolCallSession;

    use super::*;

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_sync_content_session_w_subjects() -> Result<()> {
        // Local and remote object stores
        let local_object_store_name = "Local";
        let local_object_store_backend = ObjectStorageBackend::LocalFs;
        let local_tmp_dir = TempDir::new()?;
        let local_object_store_bucket = local_tmp_dir.path().join("LocalBucketWSubject");
        let _ = std::fs::create_dir(&local_object_store_bucket);
        let remote_object_store_name = "Remote";
        let remote_object_store_backend = ObjectStorageBackend::LocalFs;
        let remote_tmp_dir = TempDir::new()?;
        let remote_object_store_bucket = remote_tmp_dir.path().join("RemoteBucketWSubject");
        let _ = std::fs::create_dir(&remote_object_store_bucket);

        // Initialize the session
        let sync_content_session = SyncContentSession {
            network_name: "sync_content_session",
            local_object_store_name,
            local_object_store_backend: local_object_store_backend.clone(),
            local_object_store_bucket: Some(local_object_store_bucket.to_str().unwrap()),
            remote_object_store_name,
            remote_object_store_backend: remote_object_store_backend.clone(),
            remote_object_store_bucket: Some(remote_object_store_bucket.to_str().unwrap()),
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &sync_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &sync_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(sync_content_session.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let subject_name = "test_subject";

        // Local data
        // DM: create mock data for deletion
        // DM, todo!(): currently, there needs to be some entries in the local metadata subject for the session to run...
        let store = make_store(
            &local_object_store_backend,
            Some(&local_object_store_bucket.to_str().unwrap().to_string()),
            None,
        )?;
        let runtime_env = RuntimeEnv::get_builder()
            .with_name("rt_local_fs")
            .with_object_store(store)
            .with_object_store_backend(&local_object_store_backend)
            .with_object_store_bucket(local_object_store_bucket.to_str().unwrap())
            .build_arc()?;
        {
            // Writes to object store
            let test_subject = test_subject::make_test_subject(subject_name, 1, 8, 1)?;
            let publication: Vec<_> = Publication::Extend {
                subject_name: subject_name.to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                test_subject.get_record_batches_own(),
                0,
                "",
                "",
            )?
            .unwrap()
            .try_collect()
            .await?;
            let messages = format!("{local_object_store_name}_meta_s");
            let message_subject = Subject::get_builder()
                .with_name(&messages)
                .with_record_batches(publication)?
                .build()?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(&messages)
                    .with_publisher(sync_content_session.network_name)
                    .with_subject(&messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_subject.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Remote data
        {
            // Writes to object store
            let test_subject = test_subject::make_test_subject(subject_name, 4, 8, 3)?;
            let store = make_store(
                &remote_object_store_backend,
                Some(&remote_object_store_bucket.to_str().unwrap().to_string()),
                None,
            )?;
            let runtime_env = RuntimeEnv::get_builder()
                .with_name("rt_remote_fs")
                .with_object_store(store)
                .with_object_store_backend(&remote_object_store_backend)
                .with_object_store_bucket(remote_object_store_bucket.to_str().unwrap())
                .build_arc()?;
            let _publication: Vec<_> = Publication::Extend {
                subject_name: subject_name.to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                test_subject.get_record_batches_own(),
                0,
                "",
                "",
            )?
            .unwrap()
            .try_collect()
            .await?;

            // Messages
            let name = format!("list_{remote_object_store_name}_p");
            let messages = format!("{remote_object_store_name}_meta_s");
            let config = ObjectStoreConfig {
                timeout: 5,
                ops_type: ObjectStoreOptsType::List,
                backend: ObjectStorageBackend::LocalFs,
                bucket: Some(remote_object_store_bucket.to_str().unwrap().to_string()),
                locations: None,
                chunk_size: None,
                subject_name: Some(messages.to_string()),
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config)?;
            let config_batch = create_bytes_record_batch(vec![config_json])?;
            let config_table = SubjectBuilder::new()
                .with_name(&name)
                .with_record_batches(vec![config_batch])?
                .build()?;
            let _ = message_map.insert(
                config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(config_table.get_name())
                    .with_publisher(sync_content_session.network_name)
                    .with_subject(config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: config_table.get_name().to_string(),
                    })
                    .with_message(config_table.to_ipc_stream()?)
                    .build()?,
            );
            let location = vec![String::new()];
            let bucket = vec![remote_object_store_bucket.to_str().unwrap().to_string()];
            let e_tag = vec![String::new()];
            let version = vec![String::new()];
            let size = vec![0_u32];
            let last_modified = vec![0_i64];
            let message_batch = create_object_store_meta_batch(
                location,
                bucket,
                e_tag,
                version,
                size,
                last_modified,
            )?;
            let message_subject = Subject::get_builder()
                .with_name(&messages)
                .with_record_batches(vec![message_batch])?
                .build()?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(&messages)
                    .with_publisher(sync_content_session.network_name)
                    .with_subject(&messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_subject.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionErrors,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionTraces.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        if !batches.is_empty() {
            let subject = Subject::get_builder()
                .with_name(AvailableSubjects::SessionTraces.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            println!(
                "{}\n{}",
                AvailableSubjects::SessionTraces,
                String::from_utf8(subject.to_csv(b',', true)?)?
            );
        }

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("list_{remote_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("list_{remote_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let mut column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        column.sort();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("RemoteBucketWSubject"));
        assert!(column.get(1).unwrap().contains("RemoteBucketWSubject"));
        assert!(column.get(2).unwrap().contains("RemoteBucketWSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [2376, 2376, 2376]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("diff_{remote_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("diff_{remote_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("diff");
        assert_eq!(column.len(), 4);
        // assert_eq!(column, ["",
        //     "{\"bucket\":\"/tmp/.tmpuYwLr1/RemoteBucketWSubject\",\"e_tag\":\"6adc7-64e2d4dc7e0d4-948\",\"last_modified\":1774806345703636,\"size\":2376,\"version\":\"\"}",
        //     "{\"bucket\":\"/tmp/.tmpuYwLr1/RemoteBucketWSubject\",\"e_tag\":\"6adc9-64e2d4dc7e0d4-948\",\"last_modified\":1774806345703636,\"size\":2376,\"version\":\"\"}",
        //     "{\"bucket\":\"/tmp/.tmpuYwLr1/RemoteBucketWSubject\",\"e_tag\":\"6adcb-64e2d4dc7f074-948\",\"last_modified\":1774806345707636,\"size\":2376,\"version\":\"\"}"
        // ]);
        let column = subject.get_column_as_vec_str("operator");
        assert_eq!(column, ["Delete", "Create", "Create", "Create"]);

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("{local_object_store_name}_meta_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("{local_object_store_name}_meta_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let mut column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        column.sort();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("RemoteBucketWSubject"));
        assert!(column.get(1).unwrap().contains("RemoteBucketWSubject"));
        assert!(column.get(2).unwrap().contains("RemoteBucketWSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [2376, 2376, 2376]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("delete_{local_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("delete_{local_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            column,
            ["session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc"]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("LocalBucketWSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 1);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 1);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [0]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("put_{local_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("put_{local_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let mut column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        column.sort();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("LocalBucketWSubject"));
        assert!(column.get(1).unwrap().contains("LocalBucketWSubject"));
        assert!(column.get(2).unwrap().contains("LocalBucketWSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [2376, 2376, 2376]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_sync_content_session_wo_subjects() -> Result<()> {
        // Local and remote object stores
        let local_object_store_name = "Local";
        let local_object_store_backend = ObjectStorageBackend::LocalFs;
        let local_tmp_dir = TempDir::new()?;
        let local_object_store_bucket = local_tmp_dir.path().join("LocalBucketWOSubject");
        let _ = std::fs::create_dir(&local_object_store_bucket);
        let remote_object_store_name = "Remote";
        let remote_object_store_backend = ObjectStorageBackend::LocalFs;
        let remote_tmp_dir = TempDir::new()?;
        let remote_object_store_bucket = remote_tmp_dir.path().join("RemoteBucketWOSubject");
        let _ = std::fs::create_dir(&remote_object_store_bucket);

        // View task session
        let tool_call_subject = format!("list_{remote_object_store_name}_p");
        let tool_call_subjects = [tool_call_subject.as_str()];
        let tool_call_session = ToolCallSession::new("tool_call_session", &tool_call_subjects);
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
        let sync_content_session = SyncContentSession {
            network_name: "sync_content_session",
            local_object_store_name,
            local_object_store_backend: local_object_store_backend.clone(),
            local_object_store_bucket: Some(local_object_store_bucket.to_str().unwrap()),
            remote_object_store_name,
            remote_object_store_backend: remote_object_store_backend.clone(),
            remote_object_store_bucket: Some(remote_object_store_bucket.to_str().unwrap()),
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &sync_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &sync_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(sync_content_session.network_name)
        .with_diagnostics(true)
        .extend(tool_call_network_builder)?
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let subject_name = "test_subject";

        // Local data
        // DM: create mock data for deletion
        // DM, todo!(): currently, there needs to be some entries in the local metadata subject for the session to run...
        let store = make_store(
            &local_object_store_backend,
            Some(&local_object_store_bucket.to_str().unwrap().to_string()),
            None,
        )?;
        let runtime_env = RuntimeEnv::get_builder()
            .with_name("rt_local_fs")
            .with_object_store(store)
            .with_object_store_backend(&local_object_store_backend)
            .with_object_store_bucket(local_object_store_bucket.to_str().unwrap())
            .build_arc()?;
        {
            // Writes to object store
            let test_subject = test_subject::make_test_subject(subject_name, 1, 8, 1)?;
            let publication: Vec<_> = Publication::Extend {
                subject_name: subject_name.to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                test_subject.get_record_batches_own(),
                0,
                "",
                "",
            )?
            .unwrap()
            .try_collect()
            .await?;
            let messages = format!("{local_object_store_name}_meta_s");
            let message_subject = Subject::get_builder()
                .with_name(&messages)
                .with_record_batches(publication)?
                .build()?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(&messages)
                    .with_publisher(sync_content_session.network_name)
                    .with_subject(&messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_subject.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Remote data
        {
            // Writes to object store
            let test_subject = test_subject::make_test_subject(subject_name, 4, 8, 3)?;
            let store = make_store(
                &remote_object_store_backend,
                Some(&remote_object_store_bucket.to_str().unwrap().to_string()),
                None,
            )?;
            let runtime_env = RuntimeEnv::get_builder()
                .with_name("rt_remote_fs")
                .with_object_store(store)
                .with_object_store_backend(&remote_object_store_backend)
                .with_object_store_bucket(remote_object_store_bucket.to_str().unwrap())
                .build_arc()?;
            let _publication: Vec<_> = Publication::Extend {
                subject_name: subject_name.to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                test_subject.get_record_batches_own(),
                0,
                "",
                "",
            )?
            .unwrap()
            .try_collect()
            .await?;

            // Messages
            let name = format!("list_{remote_object_store_name}_p");
            let config = ObjectStoreConfig {
                timeout: 5,
                ops_type: ObjectStoreOptsType::List,
                backend: ObjectStorageBackend::LocalFs,
                bucket: Some(remote_object_store_bucket.to_str().unwrap().to_string()),
                locations: Some(vec![String::new()]),
                chunk_size: None,
                subject_name: None,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config)?;
            let config_batch = create_bytes_record_batch(vec![config_json])?;
            let config_table = SubjectBuilder::new()
                .with_name(&name)
                .with_record_batches(vec![config_batch])?
                .build()?;
            let _ = message_map.insert(
                config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(config_table.get_name())
                    .with_publisher(sync_content_session.network_name)
                    .with_subject(config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: config_table.get_name().to_string(),
                    })
                    .with_message(config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        {
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::SessionErrors.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            if !batches.is_empty() {
                let subject = Subject::get_builder()
                    .with_name(AvailableSubjects::SessionErrors.to_string().as_str())
                    .with_record_batches(batches)?
                    .build()?;
                println!(
                    "{}\n{}",
                    AvailableSubjects::SessionErrors,
                    String::from_utf8(subject.to_csv(b',', true)?)?
                );
            }
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableSubjects::SessionMetrics.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            if !batches.is_empty() {
                let subject = Subject::get_builder()
                    .with_name(AvailableSubjects::SessionMetrics.to_string().as_str())
                    .with_record_batches(batches)?
                    .build()?;
                println!(
                    "{}\n{}",
                    AvailableSubjects::SessionMetrics,
                    String::from_utf8(subject.to_csv(b',', true)?)?
                );
            }
        }

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("list_{remote_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("list_{remote_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let mut column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        column.sort();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("RemoteBucketWOSubject"));
        assert!(column.get(1).unwrap().contains("RemoteBucketWOSubject"));
        assert!(column.get(2).unwrap().contains("RemoteBucketWOSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [2376, 2376, 2376]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("diff_{remote_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("diff_{remote_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("diff");
        assert_eq!(column.len(), 4);
        // assert_eq!(column, ["",
        //     "{\"bucket\":\"/tmp/.tmpuYwLr1/RemoteBucketWOSubject\",\"e_tag\":\"6adc7-64e2d4dc7e0d4-948\",\"last_modified\":1774806345703636,\"size\":2376,\"version\":\"\"}",
        //     "{\"bucket\":\"/tmp/.tmpuYwLr1/RemoteBucketWOSubject\",\"e_tag\":\"6adc9-64e2d4dc7e0d4-948\",\"last_modified\":1774806345703636,\"size\":2376,\"version\":\"\"}",
        //     "{\"bucket\":\"/tmp/.tmpuYwLr1/RemoteBucketWOSubject\",\"e_tag\":\"6adcb-64e2d4dc7f074-948\",\"last_modified\":1774806345707636,\"size\":2376,\"version\":\"\"}"
        // ]);
        let column = subject.get_column_as_vec_str("operator");
        assert_eq!(column, ["Delete", "Create", "Create", "Create"]);

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("{local_object_store_name}_meta_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("{local_object_store_name}_meta_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let mut column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        column.sort();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("RemoteBucketWOSubject"));
        assert!(column.get(1).unwrap().contains("RemoteBucketWOSubject"));
        assert!(column.get(2).unwrap().contains("RemoteBucketWOSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [2376, 2376, 2376]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("delete_{local_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("delete_{local_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            column,
            ["session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc"]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("LocalBucketWOSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 1);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 1);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [0]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: format!("put_{local_object_store_name}_s"),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(format!("put_{local_object_store_name}_s").as_str())
            .with_record_batches(batches)?
            .build()?;
        let mut column = subject
            .get_column_as_vec_str("location")
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        column.sort();
        assert_eq!(
            column,
            [
                "session=/subject=test_subject/superstep=0/publisher=/partition=0/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=1/test_subject.ipc",
                "session=/subject=test_subject/superstep=0/publisher=/partition=2/test_subject.ipc"
            ]
        );
        let column = subject.get_column_as_vec_str("bucket");
        assert!(column.first().unwrap().contains("LocalBucketWOSubject"));
        assert!(column.get(1).unwrap().contains("LocalBucketWOSubject"));
        assert!(column.get(2).unwrap().contains("LocalBucketWOSubject"));
        let column = subject.get_column_as_vec_str("e_tag");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_str("version");
        assert_eq!(column.len(), 3);
        let column = subject.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(column, [2376, 2376, 2376]);
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }

        Ok(())
    }
}
