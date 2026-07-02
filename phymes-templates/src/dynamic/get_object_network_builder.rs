use object_store::aws::AmazonS3ConfigKey;
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_network::{DynamicTaskNetworkBuilder, DynamicTaskNetworkNames, DynamicTaskNetworkNames};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{
    AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait,
    create_object_store_meta_batch,
};
use phymes_streams::{ObjectStoreConfig, ObjectStoreOptsType};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, ObjectStorageBackend, SubjectBuilder,
    SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait,
};
use serde_json::{Map, Value};

pub struct GetObjectNetworkBuilderStaticWSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for GetObjectNetworkBuilderStaticWSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "get_object";

        // Processor subject
        let mut store_config = Map::<String, Value>::new();
        let _ = store_config.insert(
            AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
            Value::String("true".to_string()),
        );
        let _ = store_config.insert(
            AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
            Value::String("https://s3.amazonaws.com".to_string()),
        );
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::Aws,
            bucket: Some("openalex".to_string()),
            backend_config: Some(serde_json::to_string(&store_config).unwrap()),
            subject_name: Some(AvailableSubjects::ObjectStoreMeta.to_string()),
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
        let subject = AvailableSubjects::ObjectStoreMeta
            .to_subject(None, None)
            .unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableInterfaceSubjects::UserObject
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
            processor: AvailableProcessors::ObjectStoreProcessor,
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

pub struct GetObjectNetworkBuilderDynamicWSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for GetObjectNetworkBuilderDynamicWSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "get_object";

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
        // let location = vec!["data/works/updated_date=2018-01-12/part_0000.gz".to_string()];
        let location = vec!["data/works/updated_date=2026-03-10/part_0005.gz".to_string()];
        // let location = vec!["data/works/manifest".to_string()];
        let bucket = vec!["openalex".to_string()];
        let e_tag = vec![String::new()];
        let version = vec![String::new()];
        let size = vec![0_u32];
        let last_modified = vec![0_i64];
        let batch =
            create_object_store_meta_batch(location, bucket, e_tag, version, size, last_modified)
                .unwrap();
        let subject = AvailableSubjects::ObjectStoreMeta
            .to_subject(None, Some(vec![batch]))
            .unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableInterfaceSubjects::UserObject
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
            processor: AvailableProcessors::ObjectStoreProcessor,
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

pub struct GetObjectNetworkBuilderDynamicWOSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for GetObjectNetworkBuilderDynamicWOSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "get_object";

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
        let subject = AvailableSubjects::ObjectStoreMeta
            .to_subject(None, None)
            .unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableInterfaceSubjects::UserObject
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
            processor: AvailableProcessors::ObjectStoreProcessor,
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
    use phymes_schemas::{
        AvailableInterfaceSubjects, AvailableSubjects, create_bytes_record_batch,
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnvBuilder, Subject,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_object_network_static_w_subject() -> Result<()> {
        let get_content_network = GetObjectNetworkBuilderStaticWSubject::default();
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
        // let location = vec!["data/works/updated_date=2018-01-12/part_0000.gz".to_string()];
        let location = vec!["data/works/updated_date=2026-03-10/part_0005.gz".to_string()];
        // let location = vec!["data/works/manifest".to_string()];
        let bucket = vec!["openalex".to_string()];
        let e_tag = vec![String::new()];
        let version = vec![String::new()];
        let size = vec![0_u32];
        let last_modified = vec![0_i64];
        let batch =
            create_object_store_meta_batch(location, bucket, e_tag, version, size, last_modified)?;
        let subject = AvailableSubjects::ObjectStoreMeta
            .to_subject(None, Some(vec![batch]))
            .unwrap();
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
                .with_message(subject.to_ipc_stream()?)
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
            subject_name: AvailableInterfaceSubjects::UserObject.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserObject.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("location");
        assert_eq!(column, ["data/works/updated_date=2026-03-10/part_0005.gz"]);
        let column = subject.get_column_as_vec_str("bucket");
        assert_eq!(column, ["openalex"]);
        let column = subject.get_column_as_vec_str("metadata");
        for c in column {
            assert!(!c.is_empty());
        }
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }
        let column = subject.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        for c in column {
            assert!(!c.is_empty());
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_object_network_dynamic_w_subject() -> Result<()> {
        let get_content_network = GetObjectNetworkBuilderDynamicWSubject::default();
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
        let mut store_config = Map::<String, Value>::new();
        let _ = store_config.insert(
            AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
            Value::String("true".to_string()),
        );
        let _ = store_config.insert(
            AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
            Value::String("https://s3.amazonaws.com".to_string()),
        );
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::Aws,
            bucket: Some("openalex".to_string()),
            backend_config: Some(serde_json::to_string(&store_config).unwrap()),
            subject_name: Some(AvailableSubjects::ObjectStoreMeta.to_string()),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config).unwrap();
        let batch = create_bytes_record_batch(vec![config_json])?;
        let subject = AvailableSubjects::Bytes.to_subject(
            Some(
                DynamicTaskNetworkNames::Processor(&get_content_network.inner.network_name)
                    .to_string()
                    .as_str(),
            ),
            Some(vec![batch]),
        )?;
        let _ = message_map.insert(
            subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(subject.get_name())
                .with_publisher(&get_content_network.inner.network_name)
                .with_subject(subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
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
            subject_name: AvailableInterfaceSubjects::UserObject.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserObject.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("location");
        assert_eq!(column, ["data/works/updated_date=2026-03-10/part_0005.gz"]);
        let column = subject.get_column_as_vec_str("bucket");
        assert_eq!(column, ["openalex"]);
        let column = subject.get_column_as_vec_str("metadata");
        for c in column {
            assert!(!c.is_empty());
        }
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }
        let column = subject.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        for c in column {
            assert!(!c.is_empty());
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_get_object_network_dynamic_wo_subject() -> Result<()> {
        let get_content_network = GetObjectNetworkBuilderDynamicWSubject::default();
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
        let mut store_config = Map::<String, Value>::new();
        let _ = store_config.insert(
            AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
            Value::String("true".to_string()),
        );
        let _ = store_config.insert(
            AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
            Value::String("https://s3.amazonaws.com".to_string()),
        );
        let location = vec!["data/works/updated_date=2026-03-10/part_0005.gz".to_string()];
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::Aws,
            bucket: Some("openalex".to_string()),
            backend_config: Some(serde_json::to_string(&store_config).unwrap()),
            subject_name: None,
            locations: Some(location),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config).unwrap();
        let batch = create_bytes_record_batch(vec![config_json])?;
        let subject = AvailableSubjects::Bytes.to_subject(
            Some(
                DynamicTaskNetworkNames::Processor(&get_content_network.inner.network_name)
                    .to_string()
                    .as_str(),
            ),
            Some(vec![batch]),
        )?;
        let _ = message_map.insert(
            subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(subject.get_name())
                .with_publisher(&get_content_network.inner.network_name)
                .with_subject(subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: subject.get_name().to_string(),
                })
                .with_message(subject.to_ipc_stream()?)
                .build()?,
        );

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
            subject_name: AvailableInterfaceSubjects::UserObject.to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(AvailableInterfaceSubjects::UserObject.to_string().as_str())
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("location");
        assert_eq!(column, ["data/works/updated_date=2026-03-10/part_0005.gz"]);
        let column = subject.get_column_as_vec_str("bucket");
        assert_eq!(column, ["openalex"]);
        let column = subject.get_column_as_vec_str("metadata");
        for c in column {
            assert!(!c.is_empty());
        }
        let column = subject.get_column_as_vec_primitive::<i64>("last_modified")?;
        for c in column {
            assert!(c > 0);
        }
        let column = subject.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        for c in column {
            assert!(!c.is_empty());
        }
        Ok(())
    }
}
