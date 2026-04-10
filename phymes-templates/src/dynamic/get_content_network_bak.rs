use phymes_subject::ObjectStorageBackend;
use phymes_streams::HTTPClientRequestSchemas;
use serde_json::{Map, Value};

/// A session for downloading PDF documents from a HTTP Request
pub struct GetContentNetwork<'a> {
    /// Session
    pub network_name: &'a str,
    /// Dynamic pipeline (e.g., tool call) or static pipeline
    pub is_dynamic: bool,
    /// Static HTTPProcessor PDF request schema
    pub pdf_request_schema: HTTPClientRequestSchemas,
    /// Static HTTPProcessor PDF base URL
    pub pdf_base_url: &'a str,
    /// Static HTTPProcessor JSON request schema
    pub json_request_schema: HTTPClientRequestSchemas,
    /// Static HTTPProcessor JSON base URL
    pub json_base_url: &'a str,
    /// Static ObjectStore backend
    pub object_store_backend: ObjectStorageBackend,
    /// Static ObjectStore bucket
    pub object_store_bucket: Option<&'a str>,
    /// Static ObjectStore config
    pub object_store_config: Option<&'a Map<String, Value>>,
}

impl<'a> Default for GetContentNetwork<'a> {
    fn default() -> Self {
        Self {
            network_name: "get_content_session",
            is_dynamic: false,
            pdf_request_schema: HTTPClientRequestSchemas::Attachments,
            pdf_base_url: "",
            json_request_schema: HTTPClientRequestSchemas::Attachments,
            json_base_url: "",
            object_store_backend: ObjectStorageBackend::default(),
            object_store_bucket: None,
            object_store_config: None,
        }
    }
}

impl<'a> GetContentNetwork<'a> {
    /// Return the Mermaid.js flowchart representation of the session
    pub fn as_mermaid_flowchart(&self) -> String {
        let network_name = self.network_name;
        let (get_pdf_p_subgraph, get_json_p_subgraph, get_object_p_subgraph) = if self.is_dynamic {
            (
                r#"
		get_pdf_p-subject-.->|LastRecordBatch|get_pdf_p-subscribe"#,
                r#"
		get_json_p-subject-.->|LastRecordBatch|get_json_p-subscribe"#,
                r#"
		get_object_p-subject-.->|LastRecordBatch|get_object_p-subscribe"#,
            )
        } else {
            ("", "", "")
        };
        let (get_pdf_p_subject, get_json_p_subject, get_object_p_subject) = if self.is_dynamic {
            (
                r#"
	get_pdf_p-subject@{shape: doc, label: get_pdf_p}"#,
                r#"
	get_json_p-subject@{shape: doc, label: get_json_p}"#,
                r#"
	get_object_p-subject@{shape: doc, label: get_object_p}"#,
            )
        } else {
            ("", "", "")
        };
        format!(
            r#"flowchart TD
	{network_name}_r-rt@{{shape: subproc, label: get_content_r}}
	%% ------------------------------------------------------------------------------
	%% PDF document downloading
    %% - We listen for updates both on the config `get_pdf_p` subject
    %%   AND a data `http_client_request_pdf_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the PDF from
    %% - The `invoke_task_network` is used to trigger the download when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph get_pdf_t
		http_client_request_pdf_s-subject-.->|AllRecordBatches|get_pdf_p-subscribe{get_pdf_p_subgraph}
		get_pdf_p-subscribe-->get_pdf_p-processor
		get_pdf_p-processor-->get_pdf_p-publish
		get_pdf_p-publish-->|Extend|UserPdf-subject
	end
	{network_name}_r-rt-->get_pdf_t
	http_client_request_pdf_s-subject@{{shape: doc, label: http_client_request_pdf_s}}{get_pdf_p_subject}
	get_pdf_p-processor@{{shape: rect, label: HTTPClientRequestProcessor}}
	get_pdf_p-publish@{{shape: fork}}
	get_pdf_p-subscribe@{{shape: diamond, label: All}}
	UserPdf-subject@{{shape: doc, label: UserPdf}}
	%% ------------------------------------------------------------------------------
	%% JSON document downloading
    %% - We listen for updates both on the config `get_json_p` subject
    %%   AND a data `http_client_request_json_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the JSON from
    %% - The `invoke_task_network` is used to trigger the download when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph get_json_t
		http_client_request_json_s-subject-.->|AllRecordBatches|get_json_p-subscribe{get_json_p_subgraph}
		get_json_p-subscribe-->get_json_p-processor
		get_json_p-processor-->get_json_p-publish
		get_json_p-publish-->|Extend|UserJson-subject
	end
	{network_name}_r-rt-->get_json_t
	http_client_request_json_s-subject@{{shape: doc, label: http_client_request_json_s}}{get_json_p_subject}
	get_json_p-processor@{{shape: rect, label: HTTPClientRequestProcessor}}
	get_json_p-publish@{{shape: fork}}
	get_json_p-subscribe@{{shape: diamond, label: All}}
	UserJson-subject@{{shape: doc, label: UserJson}}
	%% ------------------------------------------------------------------------------
	%% Object store downloading
    %% - We listen for updates both on the config `get_object_p` subject
    %%   AND a data `object_store_request_object_s` subject which is in the form of a UserMessage
    %%   which can both specify the URL to download the JSON from
    %% - The `invoke_task_network` is used to trigger the download when only the config is updated
	%% ------------------------------------------------------------------------------
	subgraph get_object_t
		object_store_request_object_s-subject-.->|AllRecordBatches|get_object_p-subscribe{get_object_p_subgraph}
		get_object_p-subscribe-->get_object_p-processor
		get_object_p-processor-->get_object_p-publish
		get_object_p-publish-->|Extend|UserObject-subject
	end
	{network_name}_r-rt-->get_object_t
	object_store_request_object_s-subject@{{shape: doc, label: object_store_request_object_s}}{get_object_p_subject}
	get_object_p-processor@{{shape: rect, label: ObjectStoreProcessor.}}
	get_object_p-publish@{{shape: fork}}
	get_object_p-subscribe@{{shape: diamond, label: All}}
	UserObject-subject@{{shape: doc, label: UserObject}}
	%% ------------------------------------------------------------------------------
    %% Next steps
	%% - Other document downloads can be added as shown above...
	%% - Other tool calls can be integrated based on the above template...
	%% - A tool message needs to be generated based on the responses...
	%% ------------------------------------------------------------------------------"#
        )
    }
    /// Return the Mermaid.js ER diagram representation of the session
    pub fn as_mermaid_erdiagram(&self) -> String {
        let (get_pdf_p, get_json_p, get_object_p) = if self.is_dynamic {
            (
                r#"
        List-UInt8 bytes"#
                    .to_string(),
                r#"
        List-UInt8 bytes"#
                    .to_string(),
                r#"
        List-UInt8 bytes"#
                    .to_string(),
            )
        } else {
            let pdf_request_schema = self.pdf_request_schema.to_string();
            let pdf_base_url = self.pdf_base_url;
            let json_request_schema = self.json_request_schema.to_string();
            let json_base_url = self.json_base_url;
            let object_store_backend = self.object_store_backend.to_string();
            let object_store_bucket = self.object_store_bucket.unwrap_or_default();
            let object_store_config = if let Some(config) = self.object_store_config {
                serde_json::to_string(config).unwrap().replace('"', "'")
            } else {
                "{}".to_string()
            };
            (
                format!(
                    r#"
        UInt32 timeout "15"
        Utf8 request_type "Get"
        Utf8 subject_name "http_client_request_pdf_s"
        Utf8 user_agent_type "rust-openalex-client/2.0"
        Utf8 request_schema "{pdf_request_schema}"
        Utf8 base_url "{pdf_base_url}""#
                ),
                format!(
                    r#"
        UInt32 timeout "15"
        Utf8 request_type "Get"
        Utf8 subject_name "http_client_request_json_s"
        Utf8 user_agent_type "rust-openalex-client/2.0"
        Utf8 request_schema "{json_request_schema}"
        Utf8 base_url "{json_base_url}""#
                ),
                format!(
                    r#"
        UInt32 timeout "15"
        Utf8 ops_type "Get"
        Utf8 backend "{object_store_backend}"
        Utf8 bucket "{object_store_bucket}"
        Utf8 backend_config "{object_store_config}"
        Utf8 subject_name "object_store_request_object_s""#
                ),
            )
        };
        format!(
            r#"erDiagram
    http_client_request_pdf_s["http_client_request_pdf_s"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    get_pdf_p["get_pdf_p"] {{{get_pdf_p}
    }}
    UserPdf["UserPdf"] {{
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }}
    http_client_request_json_s["http_client_request_json_s"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    get_json_p["get_json_p"] {{{get_json_p}
    }}
    UserJson["UserJson"] {{
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
    }}
    object_store_request_object_s["object_store_request_object_s"] {{
        Utf8 role
        Utf8 content
        Int64 timestamp
    }}
    get_object_p["get_object_p"] {{{get_object_p}
    }}
    UserObject["UserObject"] {{
        Utf8 filename
        Utf8 extension
        List-UInt8 bytes
        Utf8 metadata
        Int64 timestamp
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
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{
        NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderMermaidTrait,
        NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, create_bytes_record_batch,
    };
    use phymes_streams::{
        ChatBuilderTraitExt, HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType,
    };
    use phymes_task::SubscriptionTrait;

    use crate::InvokeTaskNetwork;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_content_session_dynamic_w_subjects() -> Result<()> {
        // Initialize the session
        let get_content_session = GetContentNetwork {
            is_dynamic: true,
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &get_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &get_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(get_content_session.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        {
            let name = "get_pdf_p";
            let messages = "http_client_request_pdf_s";
            let id = "2508.18700";
            let get_url = format!("pdf/{id}");
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
                    .with_publisher(get_content_session.network_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
                    .build()?,
            );
            let message_builder = SubjectBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&get_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(get_content_session.network_name)
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
            let name = "get_json_p";
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
                    .with_publisher(get_content_session.network_name)
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
                    .with_publisher(get_content_session.network_name)
                    .with_subject(messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
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

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(column, ["https://arxiv.org/pdf/2508.18700"]);
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
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
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
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Diabetes%20Mellitus%5BMeSH%20Terms%5D%20AND%20%22Lancet%22%5BJournal%5D&retmode=json&retmax=5&mindate=2020&maxdate=2023"
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
    async fn test_get_content_session_dynamic_wo_subjects() -> Result<()> {
        // View task session
        let invoke_task_network =
            InvokeTaskNetwork::new("invoke_task_network", &["get_pdf_p", "get_json_p"]);
        let invoke_task_network_builder = NetworkBuilder::from_mermaid_flowchart(
            &invoke_task_network.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &invoke_task_network.as_mermaid_erdiagram()?,
            false,
            true,
        )?
        .with_name(invoke_task_network.network_name);

        // Initialize the session
        let get_content_session = GetContentNetwork {
            is_dynamic: true,
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &get_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &get_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(get_content_session.network_name)
        .with_diagnostics(true)
        .extend(invoke_task_network_builder)?
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        {
            let name = "get_pdf_p";
            let id = "2508.18700";
            let get_url = format!("pdf/{id}");
            let http_client_config = HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://arxiv.org/".to_string(),
                request_schema: HTTPClientRequestSchemas::Attachments,
                json: Some(get_url),
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
                    .with_publisher(get_content_session.network_name)
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
            let name = "get_json_p";
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
                    .with_publisher(get_content_session.network_name)
                    .with_subject(http_client_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: http_client_config_table.get_name().to_string(),
                    })
                    .with_message(http_client_config_table.to_ipc_stream()?)
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

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(column, ["https://arxiv.org/pdf/2508.18700"]);
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
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
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
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Diabetes%20Mellitus%5BMeSH%20Terms%5D%20AND%20%22Lancet%22%5BJournal%5D&retmode=json&retmax=5&mindate=2020&maxdate=2023"
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
    async fn test_get_content_session_static_w_subjects() -> Result<()> {
        // Initialize the session
        let get_content_session = GetContentNetwork {
            is_dynamic: false,
            pdf_base_url: "https://arxiv.org/",
            json_base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?",
            ..Default::default()
        };
        let (network, session_messages) = NetworkBuilder::from_mermaid_flowchart(
            &get_content_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            &get_content_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(get_content_session.network_name)
        .with_diagnostics(true)
        .add_processor_subjects()?
        .add_next_tasks()?
        .add_next_supersteps()?
        .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        {
            let messages = "http_client_request_pdf_s";
            let id = "2508.18700";
            let get_url = format!("pdf/{id}");
            let message_builder = SubjectBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&get_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(get_content_session.network_name)
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
            let message_builder = SubjectBuilder::new()
                .with_name(messages)
                .append_new_user_query_str(&esearch_url, "user")?;
            let _ = message_map.insert(
                messages.to_string(),
                IPCMessage::get_builder()
                    .with_name(messages)
                    .with_publisher(get_content_session.network_name)
                    .with_subject(messages)
                    .with_update(&Publication::Replace {
                        subject_name: messages.to_string(),
                    })
                    .with_message(message_builder.clone().build()?.to_ipc_stream()?)
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

        {
            // Test supsersteps
            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(AvailableInterfaceSubjects::UserPdf.to_string().as_str())
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("filename");
            assert_eq!(column, ["https://arxiv.org/pdf/2508.18700"]);
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
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
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
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Diabetes%20Mellitus%5BMeSH%20Terms%5D%20AND%20%22Lancet%22%5BJournal%5D&retmode=json&retmax=5&mindate=2020&maxdate=2023"
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
