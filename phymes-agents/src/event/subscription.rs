use anyhow::Result;
use arrow::datatypes::Schema;
use futures::{StreamExt, TryStreamExt};
use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, DataEncoding, DataFormat, MessageBuilderTrait, ObjectStorageBackend, Publication, RecordBatchStreamAdapter, RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage, SubjectBuilder, SubjectBuilderTrait, SubjectTrait, Subscription
};
use phymes_data::{
    AvailableCandleOperators, CandleDataStream, DataConfig, DataStreamManager, LimitConfig,
    LimitStream, ObjectStoreConfig, ObjectStoreOptsType, ObjectStoreStream,
};
use phymes_diagnostics::HashMap;
use std::sync::Arc;

use crate::clear_subject;

/// List all partitions of a subject (with optional restriction to the last one)
pub fn list_subject(
    runtime_env: &Arc<RuntimeEnv>,
    session_name: &str,
    sn: &str,
    last: bool,
) -> Result<SendableRecordBatchStream> {
    // 1. List the partitions (RecordBatches)
    let location = format!("session={session_name}/subject={sn}");
    let config = ObjectStoreConfig {
        timeout: 5,
        ops_type: ObjectStoreOptsType::List,
        backend: ObjectStorageBackend::InMemory, // Force use of the runtime_env
        locations: Some(vec![location.to_string()]),
        subject_name: None,
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("ObjectStoreConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let stream = Box::pin(ObjectStoreStream::new(
        HashMap::<String, SendableRecordBatchStreamMessage>::new(),
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);

    if last {
        // 2. Sort by last_modified (Descending)
        let config = DataConfig {
            lhs_name: Some(sn.to_string()),
            lhs_values: Some(vec!["last_modified".to_string()]),
            asc: Some(false),
            cpu: false,
            operator: AvailableCandleOperators::Sort,
            lhs_stream: DataStreamManager::Accumulate,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("ObjectStoreConfig")
            .with_json(&config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            sn.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(sn)
                .with_publisher("")
                .with_subject(sn)
                .with_update(&Publication::None)
                .with_message(stream)
                .build()?,
        );
        let stream = Box::pin(CandleDataStream::new(
            message,
            config_table.to_record_batch_stream(),
            Arc::clone(runtime_env),
            None,
        )?);

        // 3. Limit to the most recent
        let config = LimitConfig {
            skip: Some(0),
            fetch: 1,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("LimitConfig")
            .with_json(&config_json, 1)?
            .build()?;
        let stream = Box::pin(LimitStream::new(
            stream,
            config_table.to_record_batch_stream(),
            Arc::clone(runtime_env),
            None,
        ));
        Ok(stream)
    } else {
        // 2. Sort by last_modified (Ascending)
        let config = DataConfig {
            lhs_name: Some(sn.to_string()),
            lhs_values: Some(vec!["last_modified".to_string()]),
            asc: Some(true),
            cpu: false,
            operator: AvailableCandleOperators::Sort,
            lhs_stream: DataStreamManager::Accumulate,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("ObjectStoreConfig")
            .with_json(&config_json, 1)?
            .build()?;
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            sn.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(sn)
                .with_publisher("")
                .with_subject(sn)
                .with_update(&Publication::None)
                .with_message(stream)
                .build()?,
        );
        let stream = Box::pin(CandleDataStream::new(
            message,
            config_table.to_record_batch_stream(),
            Arc::clone(runtime_env),
            None,
        )?);
        Ok(stream)
    }
}

/// Get all partitions of a subject in the list
pub fn get_subject(
    runtime_env: &Arc<RuntimeEnv>,
    sn: &str,
    list: SendableRecordBatchStream,
) -> Result<SendableRecordBatchStream> {
    // 4. Get all partitions (RecordBatches)
    let config = ObjectStoreConfig {
        timeout: 5,
        ops_type: ObjectStoreOptsType::Get,
        backend: ObjectStorageBackend::InMemory, // Force use of the runtime_env
        locations: None,
        subject_name: Some(sn.to_string()),
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("ObjectStoreConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(list)
            .build()?,
    );
    let stream = Box::pin(ObjectStoreStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);

    // 5. Extract the tabular subject
    let config = DataConfig {
        lhs_name: Some(sn.to_string()),
        lhs_values: Some(vec!["bytes".to_string()]),
        encoding: Some(DataEncoding::default()),
        format: Some(DataFormat::Ipc),
        schema: Some(AvailableSubjects::default()),
        cpu: false,
        operator: AvailableCandleOperators::ExtractTabular,
        lhs_stream: DataStreamManager::Stream,
        ..Default::default()
    };
    let config_json = serde_json::to_vec(&config)?;
    let config_table = SubjectBuilder::new()
        .with_name("ObjectStoreConfig")
        .with_json(&config_json, 1)?
        .build()?;
    let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
    let _ = message.insert(
        sn.to_string(),
        SendableRecordBatchStreamMessage::get_builder()
            .with_name(sn)
            .with_publisher("")
            .with_subject(sn)
            .with_update(&Publication::None)
            .with_message(stream)
            .build()?,
    );
    let stream = Box::pin(CandleDataStream::new(
        message,
        config_table.to_record_batch_stream(),
        Arc::clone(runtime_env),
        None,
    )?);
    Ok(stream)
}

/// Subscribe to a subject
pub trait SubscriptionTrait {
    /// Implement the subscription
    ///
    /// # Notes
    ///
    /// * Empty tables are skipped
    ///
    /// # Arguments
    ///
    /// * `runtime_env` - [RuntimeEnv] the object store
    fn subscribe_to_subject(
        &self,
        runtime_env: &Arc<RuntimeEnv>,
        session_name: &str,
    ) -> Result<Option<SendableRecordBatchStream>>;
}

impl SubscriptionTrait for Subscription {
    fn subscribe_to_subject(
        &self,
        runtime_env: &Arc<RuntimeEnv>,
        session_name: &str,
    ) -> Result<Option<SendableRecordBatchStream>> {
        match self {
            Self::AlwaysAllRecordBatches { subject_name: sn }
            | Self::OnUpdateAllRecordBatches { subject_name: sn } => {
                // List the partitions (RecordBatches)
                let stream = list_subject(runtime_env, session_name, sn, false)?;

                // Get all partitions (RecordBatches)
                let stream = get_subject(runtime_env, sn, stream)?;
                Ok(Some(stream))
            }
            Self::OnUpdateDrainRecordBatches { subject_name: sn } => {
                // List the partitions (RecordBatches)
                let stream = list_subject(runtime_env, session_name, sn, false)?;

                // Get all partitions (RecordBatches)
                let stream = get_subject(runtime_env, sn, stream)?;
                let schema = stream.schema(); // DM: an empty schema!

                // Clear all partitions (RecordBatches)
                let clear = clear_subject(runtime_env, session_name, sn, false)?;
                let chain = stream.chain(clear).try_filter(move |b| futures::future::ready(!b.schema().eq(&AvailableSubjects::ObjectStoreMeta.to_schema())));
                let stream = Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    chain,
                ));
                Ok(Some(stream))
            }
            Self::AlwaysLastRecordBatch { subject_name: sn }
            | Self::OnUpdateLastRecordBatch { subject_name: sn } => {
                // List the last partition (RecordBatch)
                let stream = list_subject(runtime_env, session_name, sn, true)?;

                // Get all partitions (RecordBatches)
                let stream = get_subject(runtime_env, sn, stream)?;
                Ok(Some(stream))
            }
            Self::OnUpdateEmpty { subject_name: _ } => {
                let schema = Schema::empty();
                let stream = futures::stream::iter(Vec::new().into_iter().map(Ok));
                let stream = Box::pin(RecordBatchStreamAdapter::new(Arc::new(schema), stream));
                Ok(Some(stream))
            }
            Self::None => Ok(None),
            Self::Custom(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use phymes_core::{Subject, test_subject};

    use crate::PublicationTrait;

    use super::*;

    #[tokio::test]
    async fn test_subscribe_list_subject() -> Result<()> {
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Create some dummy batches
        let messages = "messages";
        let subjects = test_subject::make_test_subject(messages, 4, 8, 3)?;

        // Write them to object storage
        let _publication: Vec<_> = Publication::Extend {
            subject_name: messages.to_string(),
        }
        .publish_to_subject(&rt_env, subjects.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;

        // List all locations
        let results: Vec<_> = list_subject(&rt_env, "", messages, false)?
            .try_collect()
            .await?;
        assert_eq!(results.len(), 1);
        let subject = Subject::get_builder()
            .with_name(messages)
            .with_record_batches(results)?
            .build()?;
        assert_eq!(subject.count_rows(), 3);
        let result = subject.get_column_as_vec_str("location");
        let result = result
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            result,
            [
                "messages/superstep=0/publisher=/partition=0/messages.ipc",
                "messages/superstep=0/publisher=/partition=1/messages.ipc",
                "messages/superstep=0/publisher=/partition=2/messages.ipc"
            ]
        );

        // List last locations
        let results: Vec<_> = list_subject(&rt_env, "", messages, true)?
            .try_collect()
            .await?;
        assert_eq!(results.len(), 1);
        let subject = Subject::get_builder()
            .with_name(messages)
            .with_record_batches(results)?
            .build()?;
        assert_eq!(subject.count_rows(), 1);
        let result = subject.get_column_as_vec_str("location");
        let result = result
            .into_iter()
            .map(|s| {
                let mut parts = s.split("-").collect::<Vec<_>>();
                let _ = parts.pop();
                parts.push(".ipc");
                parts.join("")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            result,
            ["messages/superstep=0/publisher=/partition=2/messages.ipc"]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_subscribe_last_record_batches() -> Result<()> {
        // Create some dummy batches and write them to object storage
        let subject_name = "test_table";
        let old = test_subject::make_test_subject(subject_name, 4, 0, 3)?;
        let runtime_env = Arc::new(RuntimeEnv::default());
        let _publication: Vec<_> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, old.get_record_batches_own(), 0, "", "")?
        .unwrap()
        .try_collect()
        .await?;
        let new = test_subject::make_test_subject(subject_name, 1, 0, 1)?;
        let _publication: Vec<_> = Publication::Extend {
            subject_name: subject_name.to_string(),
        }
        .publish_to_subject(&runtime_env, new.get_record_batches_own(), 1, "", "")?
        .unwrap()
        .try_collect()
        .await?;

        // Test the last RecordBatch
        let batches: Vec<_> = Subscription::AlwaysLastRecordBatch {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.get_record_batches().len(), 1);
        assert_eq!(subject.count_rows(), 1);
        let batches: Vec<_> = Subscription::OnUpdateLastRecordBatch {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.get_record_batches().len(), 1);
        assert_eq!(subject.count_rows(), 1);

        // Test all RecordBatches
        let batches: Vec<_> = Subscription::OnUpdateAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.get_record_batches().len(), 4);
        assert_eq!(subject.count_rows(), 13);

        // Test drain RecordBatches
        let batches: Vec<_> = Subscription::OnUpdateDrainRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        // let schema = batches.first().unwrap().schema();
        // let batches = batches.into_iter()
        //     .filter(|b| b.schema().eq(&schema))
        //     .collect::<Vec<_>>();
        dbg!(&batches);
        let subject = Subject::get_builder()
            .with_name(subject_name)
            .with_record_batches(batches)?
            .build()?;
        assert_eq!(subject.get_record_batches().len(), 4);
        assert_eq!(subject.count_rows(), 13);
        let batches: Vec<_> = Subscription::OnUpdateAllRecordBatches {
            subject_name: subject_name.to_string(),
        }
        .subscribe_to_subject(&runtime_env, "")?
        .unwrap()
        .try_collect()
        .await?;
        assert!(batches.is_empty());
        Ok(())
    }
}
