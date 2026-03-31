use object_store::aws::AmazonS3ConfigKey;
use serde_json::{Map, Value};

/// OpenAlex agent
pub struct OpenAlexAgentSession<'a> {
    /// Session
    pub session_context_name: &'a str,
}

impl Default for OpenAlexAgentSession<'_> {
    fn default() -> Self {
        OpenAlexAgentSession {
            session_context_name: "open_alex_agent_session",
        }
    }
}

impl<'a> OpenAlexAgentSession<'a> {
    pub fn as_mermaid_flowchart(&self) -> String {
        let session_context_name = self.session_context_name;
        format!(r#"flowchart TD
    {session_context_name}_r-rt@{{shape: subproc, label: {session_context_name}_r}}
	%% ------------------------------------------------------------------------------
	%% OpenAlex download from AWS
    %% 1. Read from the AWS bucket
    %% 2. Extract to OpenAlex Schemas
	%% ------------------------------------------------------------------------------
	subgraph get_open_alex_aws_bucket_t
		list_open_alex_aws_bucket_s-subject-.->|AllRecordBatches|get_open_alex_aws_bucket_p-subscribe
		get_open_alex_aws_bucket_p-subscribe-->get_open_alex_aws_bucket_p-processor
		get_open_alex_aws_bucket_p-processor-->get_open_alex_aws_bucket_p-publish
		get_open_alex_aws_bucket_p-publish-->|Replace|get_open_alex_aws_bucket_s-subject
	end
	{session_context_name}_r-rt-->get_open_alex_aws_bucket_t
	list_open_alex_aws_bucket_s-subject@{{shape: doc, label: list_open_alex_aws_bucket_s}}
	get_open_alex_aws_bucket_p-subscribe@{{shape: diamond, label: All}}
	get_open_alex_aws_bucket_p-processor@{{shape: rect, label: ObjectStoreProcessor}}
	get_open_alex_aws_bucket_p-publish@{{shape: fork}}
	get_open_alex_aws_bucket_s-subject@{{shape: doc, label: get_open_alex_aws_bucket_s}}
	%% ------------------------------------------------------------------------------
	subgraph extract_open_alex_aws_bucket_t
		get_open_alex_aws_bucket_s-subject-.->|AllRecordBatches|extract_open_alex_aws_bucket_p-subscribe
		extract_open_alex_aws_bucket_p-subscribe-->extract_open_alex_aws_bucket_p-processor
		extract_open_alex_aws_bucket_p-processor-->extract_open_alex_aws_bucket_p-publish
		extract_open_alex_aws_bucket_p-publish-->|Replace|extract_open_alex_aws_bucket_s-subject
		extract_open_alex_aws_bucket_p-publish-->|Replace|open_alex_works_s-subject
	end
	{session_context_name}_r-rt-->extract_open_alex_aws_bucket_t
	extract_open_alex_aws_bucket_p-subscribe@{{shape: diamond, label: All}}
	extract_open_alex_aws_bucket_p-processor@{{shape: rect, label: ExtractTabular}}
	extract_open_alex_aws_bucket_p-publish@{{shape: fork}}
	extract_open_alex_aws_bucket_s-subject@{{shape: doc, label: extract_open_alex_aws_bucket_s}}
	open_alex_works_s-subject@{{shape: doc, label: open_alex_works_s}}
	%% ------------------------------------------------------------------------------
	%% OpenAlex search for OpenAccess articles by topic
    %% 1. Filter works by Topic
    %% 2. List OpenAccess PDF URLs
	%% ------------------------------------------------------------------------------
	%% ------------------------------------------------------------------------------
	%% HTTP OpenAccess PDF download
    %% 1. Download PDF
    %% 2. ExtractPDFSession
    %% 3. EmbedTextSession (where query = Ontology term)
    %% 4. RetrieveTextSession
	%% ------------------------------------------------------------------------------
	%% ------------------------------------------------------------------------------"#)
    }
    pub fn as_mermaid_erdiagram(&self) -> String {
        let bucket = "openalex";
        let mut config = Map::<String, Value>::new();
        let _ = config.insert(
            AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
            Value::String("true".to_string()),
        );
        let _ = config.insert(
            AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
            Value::String("https://s3.amazonaws.com".to_string()),
        );
        let backend_config = serde_json::to_string(&config).unwrap().replace('"', "'");
        // List-UInt8 bytes
        format!(r#"erDiagram
    list_open_alex_aws_bucket_s["list_open_alex_aws_bucket_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 e_tag
        Utf8 version
        UInt32 size
        Int64 last_modified
    }}
    get_open_alex_aws_bucket_p["get_open_alex_aws_bucket_p"] {{
        UInt32 timeout "15"
        Utf8 ops_type "Get"
        Utf8 backend "Aws"
        Utf8 bucket "{bucket}"
        Utf8 backend_config "{backend_config}"
        Utf8 subject_name "list_open_alex_aws_bucket_s"
    }}
    get_open_alex_aws_bucket_s["get_open_alex_aws_bucket_s"] {{
        Utf8 location
        Utf8 bucket
        Utf8 metadata
        Int64 last_modified
        List-UInt8 bytes
    }}
    extract_open_alex_aws_bucket_p["extract_open_alex_aws_bucket_p"] {{
	    Boolean cpu "false"
	    Utf8 format "JsonSchema"
        Utf8 schema "OpenAlexResponseWorks"
        Utf8 encoding "Gz"
	    Utf8 lhs_name "get_open_alex_aws_bucket_s"
	    List-Utf8 lhs_values "['bytes']"
	    Utf8 operator "ExtractTabular"
	    Utf8 lhs_stream "Accumulate"
    }}
    extract_open_alex_aws_bucket_s["extract_open_alex_aws_bucket_s"] {{
        List-UInt8 bytes
    }}
    open_alex_works_s["open_alex_works_s"] {{
        Utf8 work_id
        Utf8 display_name
        Utf8 title
        Utf8 doi
        Utf8 type_
        Utf8 publication_date
        Utf8 created_date
        Utf8 updated_date
        Utf8 abstract_
        Utf8 language
        UInt32 publication_year
        UInt32 locations_count
        UInt32 countries_distinct_count
        UInt32 institutions_distinct_count
        UInt32 referenced_works_count
        Boolean is_paratext
        Boolean is_retracted
        Boolean is_xpac
    }}"#)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{
        AvailableSubjects, BuildableTrait, BuilderTrait, IPCMessage, MappableTrait, MessageBuilderTrait, ObjectStorageBackend, Publication, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait, Subscription, create_bytes_record_batch, create_object_store_meta_batch
    };
    use phymes_data::{ObjectStoreConfig, ObjectStoreOptsType};
    use phymes_diagnostics::HashMap;

    use crate::{
        SessionContextBuilder, SessionContextBuilderAgentsTrait, SessionContextBuilderMermaidTrait,
        SessionContextBuilderTrait, SessionStream, SubscriptionTrait,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_open_alex_agent_session() -> Result<()> {
        // Initialize the session
        let open_alex_agent_session = OpenAlexAgentSession::default();
        let (session_ctx, session_messages) = SessionContextBuilder::from_mermaid_flowchart(
            &open_alex_agent_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(&open_alex_agent_session.as_mermaid_erdiagram(), false, true)?
        .with_name(open_alex_agent_session.session_context_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Make the test session data
        let mut message_map = HashMap::<String, IPCMessage>::new();
        let messages = "list_open_alex_aws_bucket_s";

        // // Make the Get config
        // let bucket_name = "openalex";
        // let mut store_config = Map::<String, Value>::new();
        // let _ = store_config.insert(
        //     AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
        //     Value::String("true".to_string()),
        // );
        // let _ = store_config.insert(
        //     AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
        //     Value::String("https://s3.amazonaws.com".to_string()),
        // );
        // let config = ObjectStoreConfig {
        //     timeout: 5,
        //     ops_type: ObjectStoreOptsType::Get,
        //     backend: ObjectStorageBackend::Aws,
        //     bucket: Some(bucket_name.to_string()),
        //     backend_config: Some(store_config.clone()),
        //     locations: None,
        //     chunk_size: None,
        //     subject_name: Some(messages.to_string()),
        //     ..Default::default()
        // };
        // let config_json = serde_json::to_vec(&config)?;
        // let config_batch = create_bytes_record_batch(vec![config_json])?;
        // let config_table = SubjectBuilder::new()
        //     .with_name("get_open_alex_aws_bucket_p")
        //     .with_record_batches(vec![config_batch])?
        //     .build()?;
        // let _ = message_map.insert(
        //     config_table.get_name().to_string(),
        //     IPCMessage::get_builder()
        //         .with_name(config_table.get_name())
        //         .with_publisher(open_alex_agent_session.session_context_name)
        //         .with_subject(config_table.get_name())
        //         .with_update(&Publication::Replace {
        //             subject_name: config_table.get_name().to_string(),
        //         })
        //         .with_message(config_table.to_ipc_stream()?)
        //         .build()?,
        // );

        // Make the list of paths to Get
        // let location = vec!["data/works/updated_date=2018-01-12/part_0000.gz".to_string()];
        let location = vec!["data/works/updated_date=2026-03-10/part_0005.gz".to_string()];
        // let location = vec!["data/works/manifest".to_string()];
        let bucket = vec!["openalex".to_string()];
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
                .with_publisher(open_alex_agent_session.session_context_name)
                .with_subject(&messages)
                .with_update(&Publication::Replace {
                    subject_name: messages.to_string(),
                })
                .with_message(message_subject.to_ipc_stream()?)
                .build()?,
        );

       
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Run the session
        let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
        let response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: AvailableSubjects::SessionErrors.to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
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
        .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
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

        // Test session stream
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "open_alex_works_s".to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("open_alex_works_s")
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();

        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "extract_open_alex_aws_bucket_s".to_string(),
        }
        .subscribe_to_subject(session_ctx_arc.runtime_env(), session_ctx_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("extract_open_alex_aws_bucket_s")
            .with_record_batches(batches)
            .unwrap()
            .build()
            .unwrap();

        Ok(())
    }
}
