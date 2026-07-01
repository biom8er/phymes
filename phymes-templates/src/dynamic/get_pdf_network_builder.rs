use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait};
use phymes_network::{DynamicTaskNetworkBuilder, DynamicTaskNetworkNames};
use phymes_streams::{
    ChatBuilderTraitExt, HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};

pub struct GetPdfNetworkBuilderStaticWSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for GetPdfNetworkBuilderStaticWSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "get_pdf";
        let subject_name_lhs = "http_client_request_pdf_s";

        // Processor subject
        let config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            poll_error: true,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://arxiv.org/".to_string(),
            subject_name: Some(subject_name_lhs.to_string()),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config).unwrap();
        let subject = SubjectBuilder::new()
            .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
            .with_json(&config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Subscriptions and publications
        let subject = AvailableInterfaceSubjects::UserMessages
            .to_subject(Some(subject_name_lhs), None)
            .unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableInterfaceSubjects::UserPdf
            .to_subject(None, None)
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Initialize the network
        let builder = DynamicTaskNetworkBuilder {
            network_name: network_name.to_string(),
            dynamic_type: DynamicTaskNetworkTypes::Static,
            processor: AvailableProcessors::HTTPClientRequestProcessor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_lhs.get_name().to_string(),
            },
            publication: Publication::Extend {
                subject_name: subject_out.get_name().to_string(),
            },
            subject_lhs: Some(subject_lhs),
            subject_out: Some(subject_out),
            subject_processor,
            ..Default::default()
        };

        Self { inner: builder }
    }
}

pub struct GetPdfNetworkBuilderDynamicWSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for GetPdfNetworkBuilderDynamicWSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "get_pdf";
        let subject_name_lhs = "http_client_request_pdf_s";

        // Processor subject
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&DynamicTaskNetworkNames::Processor(network_name).to_string()),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Subscriptions and publications
        let id = "2508.18700";
        let get_url = format!("pdf/{id}");
        let subject = SubjectBuilder::new()
            .with_name(subject_name_lhs)
            .append_new_user_query_str(&get_url, "user")
            .unwrap()
            .build()
            .unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableInterfaceSubjects::UserPdf
            .to_subject(None, None)
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Initialize the network
        let builder = DynamicTaskNetworkBuilder {
            network_name: network_name.to_string(),
            dynamic_type: DynamicTaskNetworkTypes::Dynamic,
            processor: AvailableProcessors::HTTPClientRequestProcessor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_lhs.get_name().to_string(),
            },
            publication: Publication::Extend {
                subject_name: subject_out.get_name().to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_lhs: Some(subject_lhs),
            subject_out: Some(subject_out),
            subject_processor,
            ..Default::default()
        };

        Self { inner: builder }
    }
}

pub struct GetPdfNetworkBuilderDynamicWOSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for GetPdfNetworkBuilderDynamicWOSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "get_pdf";
        let subject_name_lhs = "http_client_request_pdf_s";

        // Processor subject
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&DynamicTaskNetworkNames::Processor(network_name).to_string()),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Subscriptions and publications
        let subject = AvailableInterfaceSubjects::UserMessages
            .to_subject(Some(subject_name_lhs), None)
            .unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableInterfaceSubjects::UserPdf
            .to_subject(None, None)
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Initialize the network
        let builder = DynamicTaskNetworkBuilder {
            network_name: network_name.to_string(),
            dynamic_type: DynamicTaskNetworkTypes::Dynamic,
            processor: AvailableProcessors::HTTPClientRequestProcessor,
            subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_lhs.get_name().to_string(),
            },
            publication: Publication::Extend {
                subject_name: subject_out.get_name().to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_lhs: Some(subject_lhs),
            subject_out: Some(subject_out),
            subject_processor,
            ..Default::default()
        };

        Self { inner: builder }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream};
    use phymes_schemas::{AvailableInterfaceSubjects, create_bytes_record_batch};
    use phymes_streams::{
        ChatBuilderTraitExt, HTTPClientConfig, HTTPClientRequestSchemas, HTTPClientRequestType,
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnvBuilder, Subject, SubjectBuilder,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_pdf_network_static_w_subject() -> Result<()> {
        let get_content_network = GetPdfNetworkBuilderStaticWSubject::default();
        let (network, session_messages) = get_content_network
            .inner
            .build_dynamic()
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        &DynamicTaskNetworkNames::Task(&get_content_network.inner.network_name)
                            .to_string(),
                    )
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        let id = "2508.18700";
        let get_url = format!("pdf/{id}");
        let message_builder = SubjectBuilder::new()
            .with_name(get_content_network.inner.subscription_lhs.subject_name())
            .append_new_user_query_str(&get_url, "user")?;
        let _ = message_map.insert(
            get_content_network
                .inner
                .subscription_lhs
                .subject_name()
                .to_string(),
            IPCMessage::get_builder()
                .with_name(get_content_network.inner.subscription_lhs.subject_name())
                .with_publisher(&get_content_network.inner.network_name)
                .with_subject(get_content_network.inner.subscription_lhs.subject_name())
                .with_update(&Publication::Replace {
                    subject_name: get_content_network
                        .inner
                        .subscription_lhs
                        .subject_name()
                        .to_string(),
                })
                .with_message(message_builder.clone().build()?.to_ipc_stream()?)
                .build()?,
        );

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

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
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_pdf_network_dynamic_w_subject() -> Result<()> {
        let get_content_network = GetPdfNetworkBuilderDynamicWSubject::default();
        let (network, session_messages) = get_content_network
            .inner
            .build_dynamic()
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        &DynamicTaskNetworkNames::Task(&get_content_network.inner.network_name)
                            .to_string(),
                    )
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            poll_error: true,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://arxiv.org/".to_string(),
            subject_name: Some(
                get_content_network
                    .inner
                    .subscription_lhs
                    .subject_name()
                    .to_string(),
            ),
            request_schema: HTTPClientRequestSchemas::Attachments,
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_batch = create_bytes_record_batch(vec![http_client_config_json])?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(
                &DynamicTaskNetworkNames::Processor(&get_content_network.inner.network_name)
                    .to_string(),
            )
            .with_record_batches(vec![http_client_config_batch])?
            .build()?;
        let _ = message_map.insert(
            http_client_config_table.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher(&get_content_network.inner.network_name)
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::Replace {
                    subject_name: http_client_config_table.get_name().to_string(),
                })
                .with_message(http_client_config_table.to_ipc_stream()?)
                .build()?,
        );

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

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
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_pdf_network_dynamic_wo_subject() -> Result<()> {
        let get_content_network = GetPdfNetworkBuilderDynamicWSubject::default();
        let (network, session_messages) = get_content_network
            .inner
            .build_dynamic()
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        &DynamicTaskNetworkNames::Task(&get_content_network.inner.network_name)
                            .to_string(),
                    )
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // PDF download data
        let id = "2508.18700";
        let get_url = format!("pdf/{id}");
        let http_client_config = HTTPClientConfig {
            timeout: 5,
            request_type: HTTPClientRequestType::Get,
            poll_error: true,
            user_agent_type: Some("rust-openalex-client/2.0".to_string()),
            base_url: "https://arxiv.org/".to_string(),
            subject_name: None,
            request_schema: HTTPClientRequestSchemas::Attachments,
            json: Some(get_url),
            ..Default::default()
        };
        let http_client_config_json = serde_json::to_vec(&http_client_config)?;
        let http_client_config_batch = create_bytes_record_batch(vec![http_client_config_json])?;
        let http_client_config_table = SubjectBuilder::new()
            .with_name(get_content_network.inner.subject_processor.get_name())
            .with_record_batches(vec![http_client_config_batch])?
            .build()?;
        let _ = message_map.insert(
            http_client_config_table.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(http_client_config_table.get_name())
                .with_publisher(&get_content_network.inner.network_name)
                .with_subject(http_client_config_table.get_name())
                .with_update(&Publication::Replace {
                    subject_name: http_client_config_table.get_name().to_string(),
                })
                .with_message(http_client_config_table.to_ipc_stream()?)
                .build()?,
        );

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

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
        Ok(())
    }
}
