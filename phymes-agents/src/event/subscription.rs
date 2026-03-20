use anyhow::Result;
use arrow::datatypes::Schema;
use phymes_data::{AvailableCandleOperators, CandleDataStream, DataConfig, DataStreamManager, LimitConfig, LimitStream, ObjectStoreConfig, ObjectStoreOptsType, ObjectStoreStream};
use phymes_diagnostics::HashMap;
use phymes_core::{BuildableTrait, BuilderTrait, DataEncoding, DataFormat, MessageBuilderTrait, ObjectStorageBackend, Publication, RecordBatchStreamAdapter, RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage, SubjectBuilder, SubjectBuilderTrait, SubjectTrait, Subscription};
use std::sync::Arc;

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
    fn subscribe_to_subject(&self, runtime_env: &Arc<RuntimeEnv>) -> Result<Option<SendableRecordBatchStream>>;
}

impl SubscriptionTrait for Subscription {
    fn subscribe_to_subject(
        &self,
        runtime_env: &Arc<RuntimeEnv>,
    ) -> Result<Option<SendableRecordBatchStream>> {
        match self {
            Self::AlwaysAllRecordBatches { subject_name: sn } 
            | Self::OnUpdateAllRecordBatches { subject_name: sn } => {
                // 1. List the partitions (RecordBatches)
                let config = ObjectStoreConfig {
                    timeout: 5,
                    ops_type: ObjectStoreOptsType::List,
                    backend: ObjectStorageBackend::InMemory, // Force use of the runtime_env
                    locations: Some(vec![sn.to_string()]),
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
                    Arc::clone(&runtime_env),
                    None
                )?);

                // 2. Get all partitions (RecordBatches)
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
                        .with_message(stream)
                        .build()?,
                ); 
                let stream = Box::pin(ObjectStoreStream::new(
                    message,
                    config_table.to_record_batch_stream(),
                    Arc::clone(&runtime_env),
                    None
                )?);

                // 3. Extract the tabular subject
                let config = DataConfig {
                    lhs_name: Some(sn.to_string()),
                    lhs_values: Some(vec!["bytes".to_string()]),
                    encoding: Some(DataEncoding::default()),
                    format: Some(DataFormat::Ipc),
                    schema: None,
                    cpu: false,
                    operator: AvailableCandleOperators::ExtractTabular,
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
                    Arc::clone(&runtime_env),
                    None
                )?);
                Ok(Some(stream))
            }
            Self::AlwaysLastRecordBatch { subject_name: sn }
            | Self::OnUpdateLastRecordBatch { subject_name: sn } => {
                // 1. List the partitions (RecordBatches)
                let config = ObjectStoreConfig {
                    timeout: 5,
                    ops_type: ObjectStoreOptsType::List,
                    backend: ObjectStorageBackend::InMemory, // Force use of the runtime_env
                    locations: Some(vec![sn.to_string()]),
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
                    Arc::clone(&runtime_env),
                    None
                )?);

                // 2. Sort by last_modified
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
                    Arc::clone(&runtime_env),
                    None
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
                    Arc::clone(&runtime_env),
                    None
                ));

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
                        .with_message(stream)
                        .build()?,
                ); 
                let stream = Box::pin(ObjectStoreStream::new(
                    message,
                    config_table.to_record_batch_stream(),
                    Arc::clone(&runtime_env),
                    None
                )?);

                // 5. Extract the tabular subject
                let config = DataConfig {
                    lhs_name: Some(sn.to_string()),
                    lhs_values: Some(vec!["bytes".to_string()]),
                    encoding: Some(DataEncoding::default()),
                    format: Some(DataFormat::Ipc),
                    schema: None,
                    cpu: false,
                    operator: AvailableCandleOperators::ExtractTabular,
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
                    Arc::clone(&runtime_env),
                    None
                )?);
                Ok(Some(stream))
            }
            Self::OnUpdateEmpty { subject_name: _ } => {
                let schema = Schema::empty();
                let stream = futures::stream::iter(Vec::new().into_iter().map(Ok));
                let stream = Box::pin(RecordBatchStreamAdapter::new(
                    Arc::new(schema),
                    stream,
                ));
                Ok(Some(stream))
            }
            Self::None => Ok(None),
            Self::Custom(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_to_subject() -> Result<()> {

        Ok(())
    }
}
