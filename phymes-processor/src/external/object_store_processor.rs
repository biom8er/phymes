use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::ObjectStoreStream;

use crate::ProcessorTrait;

#[derive(Debug)]
pub struct ObjectStoreProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for ObjectStoreProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for ObjectStoreProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn line_and_file(&self) -> (u32, String) {
        (line!(), file!().to_string())
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        println!("Starting processor {}", self.get_name());
        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the stream
        let out = Box::pin(ObjectStoreStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.cloned(),
        )?);

        // Prepare the message builder
        let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let builder = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.get_name())
            .with_message(out);
        let _ = builder_map.insert(self.get_name().to_string(), builder);

        Ok(builder_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use phymes_subject::{
        ObjectStorageBackend, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
        test_subject,
    };
    use phymes_diagnostics::{
        DiagnosticBuilder, DiagnosticBuilderTrait, Diagnostics, HashMap, SpanBuilder,
    };
    use phymes_event::Publication;

    use phymes_schemas::{create_object_store_batch, create_object_store_meta_batch};
    use phymes_streams::{ObjectStoreConfig, ObjectStoreOptsType};
    use serde_json::{Map, Value};
    #[cfg(not(target_family = "wasm"))]
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_object_store_processor_put_get_in_memory() -> Result<()> {
        let name = "ObjectStoreProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // PUT from messages
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Put,
            backend: ObjectStorageBackend::InMemory,
            bucket: None,
            locations: None,
            chunk_size: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Make the object store batch
        let location = ["location_1.ipc", "location_2.ipc"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let bucket = ["bucket", "bucket"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let last_modified = (0..2).map(|i| i as i64).collect::<Vec<_>>();
        let metadata = ["etag_1", "etag_2"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let bytes = vec![
            test_subject::make_test_subject("location_1", 3, 4, 2)?.to_ipc_stream()?,
            test_subject::make_test_subject("location_2", 3, 0, 2)?.to_ipc_stream()?,
        ];
        let batch = create_object_store_batch(
            location.clone(),
            bucket.clone(),
            metadata.clone(),
            last_modified.clone(),
            bytes,
        )?;
        let table = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![batch])?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("version");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(result, [4104, 3336]);
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }

        // GET from messages
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::InMemory,
            bucket: None,
            locations: None,
            chunk_size: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Make the object store meta batch
        let size = (0..2).map(|i| i as u32).collect::<Vec<_>>();
        let version = ["version_1", "version_2"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let batch = create_object_store_meta_batch(
            location.clone(),
            bucket,
            metadata,
            version,
            size,
            last_modified,
        )?;
        let table = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![batch])?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(
            result,
            [
                "{\"e_tag\":\"0\",\"size\":4104,\"version\":\"\"}",
                "{\"e_tag\":\"1\",\"size\":3336,\"version\":\"\"}"
            ]
        );
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }
        let result = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let tables: Result<Vec<Subject>> = result
            .into_iter()
            .map(|bytes| {
                SubjectBuilder::new_from_ipc_stream(&bytes)?
                    .with_name("IPC")
                    .build()
            })
            .collect();
        let tables = tables?;
        assert_eq!(
            tables.first().unwrap(),
            &test_subject::make_test_subject("IPC", 3, 4, 2)?
        );
        assert_eq!(
            tables.get(1).unwrap(),
            &test_subject::make_test_subject("IPC", 3, 0, 2)?
        );

        // GET from config
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::InMemory,
            bucket: None,
            locations: Some(location.clone()),
            chunk_size: None,
            subject_name: None,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("metadata");
        assert_eq!(
            result,
            [
                "{\"e_tag\":\"0\",\"size\":4104,\"version\":\"\"}",
                "{\"e_tag\":\"1\",\"size\":3336,\"version\":\"\"}"
            ]
        );
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }
        let result = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let tables: Result<Vec<Subject>> = result
            .into_iter()
            .map(|bytes| {
                SubjectBuilder::new_from_ipc_stream(&bytes)?
                    .with_name("IPC")
                    .build()
            })
            .collect();
        let tables = tables?;
        assert_eq!(
            tables.first().unwrap(),
            &test_subject::make_test_subject("IPC", 3, 4, 2)?
        );
        assert_eq!(
            tables.get(1).unwrap(),
            &test_subject::make_test_subject("IPC", 3, 0, 2)?
        );

        // List from config
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::List,
            backend: ObjectStorageBackend::InMemory,
            bucket: None,
            locations: Some(vec![String::new()]),
            chunk_size: None,
            subject_name: None,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("List from config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("version");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(result, [4104, 3336]);
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }

        // Delete from config
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Delete,
            backend: ObjectStorageBackend::InMemory,
            bucket: None,
            locations: Some(location),
            chunk_size: None,
            subject_name: None,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("Delete from config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_str("version");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(result, [0, 0]);
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }

        // Confirm the deletion from config
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::List,
            backend: ObjectStorageBackend::InMemory,
            bucket: None,
            locations: Some(vec![String::new()]),
            chunk_size: None,
            subject_name: None,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(result.len(), 0);

        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test]
    async fn test_object_store_processor_put_get_local_fs_messages() -> Result<()> {
        let name = "ObjectStoreProcessor";
        let messages = "messages";

        // Create project directory
        let bucket_name = "phymes-object-store";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(bucket_name);
        let _ = std::fs::create_dir(&project_dir);

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // PUT from messages
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Put,
            backend: ObjectStorageBackend::LocalFs,
            bucket: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            locations: None,
            chunk_size: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Make the object store batch
        let location = ["location_1.ipc", "location_2.ipc"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let bucket = vec![
            project_dir.clone().as_path().to_str().unwrap().to_string(),
            project_dir.clone().as_path().to_str().unwrap().to_string(),
        ];
        let last_modified = (0..2).map(|i| i as i64).collect::<Vec<_>>();
        let metadata = ["etag_1", "etag_2"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let bytes = vec![
            test_subject::make_test_subject("location_1", 3, 4, 2)?.to_ipc_stream()?,
            test_subject::make_test_subject("location_2", 3, 0, 2)?.to_ipc_stream()?,
        ];
        let batch = create_object_store_batch(
            location.clone(),
            bucket.clone(),
            metadata.clone(),
            last_modified.clone(),
            bytes,
        )?;
        let table = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![batch])?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert!(result.first().unwrap().contains("phymes-object-store"));
        assert!(result.get(1).unwrap().contains("phymes-object-store"));
        let result = table.get_column_as_vec_str("version");
        assert_eq!(result, ["", ""]);
        let result = table.get_column_as_vec_primitive::<u32>("size")?;
        assert_eq!(result, [4104, 3336]);
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }

        // GET from messages
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::LocalFs,
            bucket: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            locations: None,
            chunk_size: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Make the object store meta batch
        let size = (0..2).map(|i| i as u32).collect::<Vec<_>>();
        let version = ["version_1", "version_2"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let batch = create_object_store_meta_batch(
            location.clone(),
            bucket,
            metadata,
            version,
            size,
            last_modified,
        )?;
        let table = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![batch])?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert!(result.first().unwrap().contains("phymes-object-store"));
        assert!(result.get(1).unwrap().contains("phymes-object-store"));
        let result = table.get_column_as_vec_str("metadata");
        let metadata: Result<Vec<Map<String, Value>>> = result
            .into_iter()
            .map(|j| serde_json::from_str::<Map<String, Value>>(j).map_err(|e| e.into()))
            .collect();
        let metadata = metadata?;
        assert!(metadata.first().unwrap().get("e_tag").is_some());
        assert!(metadata.first().unwrap().get("version").is_some());
        assert_eq!(
            metadata
                .first()
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap(),
            4104
        );
        assert!(metadata.get(1).unwrap().get("e_tag").is_some());
        assert!(metadata.get(1).unwrap().get("version").is_some());
        assert_eq!(
            metadata
                .get(1)
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap(),
            3336
        );
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }
        let result = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let tables: Result<Vec<Subject>> = result
            .into_iter()
            .map(|bytes| {
                SubjectBuilder::new_from_ipc_stream(&bytes)?
                    .with_name("IPC")
                    .build()
            })
            .collect();
        let tables = tables?;
        assert_eq!(
            tables.first().unwrap(),
            &test_subject::make_test_subject("IPC", 3, 4, 2)?
        );
        assert_eq!(
            tables.get(1).unwrap(),
            &test_subject::make_test_subject("IPC", 3, 0, 2)?
        );

        // GET from config
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::LocalFs,
            bucket: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            locations: Some(location),
            chunk_size: None,
            subject_name: None,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["location_1.ipc", "location_2.ipc"]);
        let result = table.get_column_as_vec_str("bucket");
        assert!(result.first().unwrap().contains("phymes-object-store"));
        assert!(result.get(1).unwrap().contains("phymes-object-store"));
        let result = table.get_column_as_vec_str("metadata");
        let metadata: Result<Vec<Map<String, Value>>> = result
            .into_iter()
            .map(|j| serde_json::from_str::<Map<String, Value>>(j).map_err(|e| e.into()))
            .collect();
        let metadata = metadata?;
        assert!(metadata.first().unwrap().get("e_tag").is_some());
        assert!(metadata.first().unwrap().get("version").is_some());
        assert_eq!(
            metadata
                .first()
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap(),
            4104
        );
        assert!(metadata.get(1).unwrap().get("e_tag").is_some());
        assert!(metadata.get(1).unwrap().get("version").is_some());
        assert_eq!(
            metadata
                .get(1)
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap(),
            3336
        );
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }
        let result = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let tables: Result<Vec<Subject>> = result
            .into_iter()
            .map(|bytes| {
                SubjectBuilder::new_from_ipc_stream(&bytes)?
                    .with_name("IPC")
                    .build()
            })
            .collect();
        let tables = tables?;
        assert_eq!(
            tables.first().unwrap(),
            &test_subject::make_test_subject("IPC", 3, 4, 2)?
        );
        assert_eq!(
            tables.get(1).unwrap(),
            &test_subject::make_test_subject("IPC", 3, 0, 2)?
        );

        Ok(())
    }

    #[cfg(feature = "api")]
    #[tokio::test]
    async fn test_object_store_processor_get_aws_config() -> Result<()> {
        use object_store::aws::AmazonS3ConfigKey;

        let name = "ObjectStoreProcessor";
        let messages = "messages";

        // AWS S3 configs
        let bucket_name = "openalex";
        let mut store_config = Map::<String, Value>::new();
        let _ = store_config.insert(
            AmazonS3ConfigKey::SkipSignature.as_ref().to_string(),
            Value::String("true".to_string()),
        );
        let _ = store_config.insert(
            AmazonS3ConfigKey::Endpoint.as_ref().to_string(),
            Value::String("https://s3.amazonaws.com".to_string()),
        );
        // let _ = store_config.insert(AmazonS3ConfigKey::DefaultRegion.as_ref().to_string(), Value::String("true".to_string()));

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // GET from messages
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::Aws,
            bucket: Some(bucket_name.to_string()),
            backend_config: Some(serde_json::to_string(&store_config)?),
            locations: None,
            chunk_size: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Make the object store meta batch
        let location = ["data/authors/manifest", "RELEASE_NOTES.txt"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let bucket = [bucket_name, bucket_name]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let metadata = ["", ""]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let last_modified = (0..2).map(|_| 0_i64).collect::<Vec<_>>();
        let size = (0..2).map(|_| 0_u32).collect::<Vec<_>>();
        let version = ["", ""]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let batch = create_object_store_meta_batch(
            location.clone(),
            bucket,
            metadata,
            version,
            size,
            last_modified,
        )?;
        let table = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![batch])?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["data/authors/manifest", "RELEASE_NOTES.txt"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, [bucket_name, bucket_name]);
        let result = table.get_column_as_vec_str("metadata");
        let metadata: Result<Vec<Map<String, Value>>> = result
            .into_iter()
            .map(|j| serde_json::from_str::<Map<String, Value>>(j).map_err(|e| e.into()))
            .collect();
        let metadata = metadata?;
        assert!(metadata.first().unwrap().get("e_tag").is_some());
        assert!(metadata.first().unwrap().get("version").is_some());
        assert!(
            metadata
                .first()
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap()
                > 0
        );
        assert!(metadata.get(1).unwrap().get("e_tag").is_some());
        assert!(metadata.get(1).unwrap().get("version").is_some());
        assert!(
            metadata
                .get(1)
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap()
                > 0
        );
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }
        let result = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let files: Result<Vec<String>> = result
            .into_iter()
            .map(|bytes| String::from_utf8(bytes).map_err(|err| err.into()))
            .collect();
        let files = files?;
        assert!(files.first().unwrap().contains("entries"));
        assert!(files.first().unwrap().contains("meta"));
        assert!(files.first().unwrap().contains("content_length"));
        assert!(files.first().unwrap().contains("record_count"));
        assert!(
            files
                .get(1)
                .unwrap()
                .contains("OPENALEX STANDARD-FORMAT SNAPSHOT RELEASE NOTES")
        );

        // GET from config
        // Config for the Processor
        let config = ObjectStoreConfig {
            timeout: 5,
            ops_type: ObjectStoreOptsType::Get,
            backend: ObjectStorageBackend::Aws,
            bucket: Some(bucket_name.to_string()),
            backend_config: Some(serde_json::to_string(&store_config)?),
            locations: Some(location),
            chunk_size: None,
            subject_name: None,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(config_table.get_name())
                .with_publisher("")
                .with_subject(config_table.get_name())
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the processor
        let processor = ObjectStoreProcessor::new(name, ObjectStoreProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("location");
        assert_eq!(result, ["data/authors/manifest", "RELEASE_NOTES.txt"]);
        let result = table.get_column_as_vec_str("bucket");
        assert_eq!(result, [bucket_name, bucket_name]);
        let result = table.get_column_as_vec_str("metadata");
        let metadata: Result<Vec<Map<String, Value>>> = result
            .into_iter()
            .map(|j| serde_json::from_str::<Map<String, Value>>(j).map_err(|e| e.into()))
            .collect();
        let metadata = metadata?;
        assert!(metadata.first().unwrap().get("e_tag").is_some());
        assert!(metadata.first().unwrap().get("version").is_some());
        assert!(
            metadata
                .first()
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap()
                > 0
        );
        assert!(metadata.get(1).unwrap().get("e_tag").is_some());
        assert!(metadata.get(1).unwrap().get("version").is_some());
        assert!(
            metadata
                .get(1)
                .unwrap()
                .get("size")
                .unwrap()
                .as_i64()
                .unwrap()
                > 0
        );
        let result = table.get_column_as_vec_primitive::<i64>("last_modified")?;
        for res in result {
            assert!(res > 0);
        }
        let result = table.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let files: Result<Vec<String>> = result
            .into_iter()
            .map(|bytes| String::from_utf8(bytes).map_err(|err| err.into()))
            .collect();
        let files = files?;
        assert!(files.first().unwrap().contains("entries"));
        assert!(files.first().unwrap().contains("meta"));
        assert!(files.first().unwrap().contains("content_length"));
        assert!(files.first().unwrap().contains("record_count"));
        assert!(
            files
                .get(1)
                .unwrap()
                .contains("OPENALEX STANDARD-FORMAT SNAPSHOT RELEASE NOTES")
        );

        Ok(())
    }
}
