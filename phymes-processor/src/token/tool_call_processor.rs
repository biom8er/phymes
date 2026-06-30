use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::ToolCallStream;
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use tracing::{Level, event, instrument};

use crate::ProcessorTrait;

/// Processor that parses a [ProcessorTrait] configuration subject and
///   creates an on-the-fly `SessionTasksSubscribePublish` subject which calls
///   the [ProcessorTrait] with subscriptions provided in the configuration subject
///
/// # Notes
///
/// - This processor MUST subscribe to a `ViewTasksSubscribePublishAggregated` subject
/// - It is assumed that the name of the configuration subject is the SAME as the processor
#[derive(Debug)]
pub struct ToolCallProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for ToolCallProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for ToolCallProcessor {
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

    #[instrument(skip(self, message, diagnostic_builder, runtime_env))]
    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(ToolCallStream::new(
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
    use phymes_data::DataConfig;
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_event::Publication;
    use phymes_schemas::{
        AvailableSubjects, create_bytes_record_batch, create_session_tasks_subscribe_publish_batch,
    };
    use phymes_streams::ToolCallConfig;
    use phymes_subject::{Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};

    use super::*;

    #[tokio::test]
    async fn test_tool_call_processor_from_struct() -> Result<()> {
        let name = "tool_call_processor";

        // Make the diagnostics and runtime env
        let span = SpanBuilder::default().with_span(name).build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Make the tool_call_processor config
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let tool_call_processor_config = ToolCallConfig {
            subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            subject_names: vec!["processor_1".to_string(), "processor_2".to_string()],
            subscription_table_names: vec!["lhs_name".to_string(), "rhs_name".to_string()],
            ..Default::default()
        };
        let tool_call_processor_config_json = serde_json::to_vec(&tool_call_processor_config)?;
        let tool_call_processor_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&tool_call_processor_config_json, 1)?
            .build()?;
        let _ = message.insert(
            tool_call_processor_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(tool_call_processor_config_table.get_name())
                .with_publisher("")
                .with_subject(tool_call_processor_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(tool_call_processor_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the dummy processor configs
        let processor_1_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            ..Default::default()
        };
        let processor_1_config_json = serde_json::to_vec(&processor_1_config)?;
        let processor_1_config_table = SubjectBuilder::new()
            .with_name("processor_1")
            .with_json(&processor_1_config_json, 1)?
            .build()?;
        let _ = message.insert(
            processor_1_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_1_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_1_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_1_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor_2_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            ..Default::default()
        };
        let processor_2_config_json = serde_json::to_vec(&processor_2_config)?;
        let processor_2_config_table = SubjectBuilder::new()
            .with_name("processor_2")
            .with_json(&processor_2_config_json, 1)?
            .build()?;
        let _ = message.insert(
            processor_2_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_2_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_2_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_2_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the mock subject_name table
        let task_names = vec!["task_1", "task_2", "task_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Join", "Filter"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["state_1", "processor_1"],
            vec!["state_2", "processor_2"],
            vec!["state_3", "processor_3"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"], vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["state_1"], vec!["state_2"], vec!["state_3"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| "session_1".to_string())
            .collect::<Vec<_>>();
        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
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

        // Create the processor and run
        let processor = ToolCallProcessor::new(name, ToolCallProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let table_reading = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream.remove(name).unwrap().message.take().unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, ["session_1", "session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, ["processor_1", "processor_2"]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, ["GroupBy", "Join"]);
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "OnUpdateAllRecordBatches",
                "AlwaysLastRecordBatch",
                "AlwaysAllRecordBatches",
                "OnUpdateAllRecordBatches",
                "AlwaysLastRecordBatch"
            ]
        );
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "state_1",
                "processor_1",
                "state_1",
                "state_2",
                "processor_2"
            ]
        );
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["Replace", "Replace"]);
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["state_1", "state_2"]);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_call_processor_from_bytes() -> Result<()> {
        let name = "tool_call_processor";

        // Make the diagnostics and runtime env
        let span = SpanBuilder::default().with_span(name).build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Make the tool_call_processor config
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let tool_call_processor_config = ToolCallConfig {
            subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            subject_names: vec!["processor_1".to_string(), "processor_2".to_string()],
            subscription_table_names: vec!["lhs_name".to_string(), "rhs_name".to_string()],
            ..Default::default()
        };
        let tool_call_processor_config_json = serde_json::to_vec(&tool_call_processor_config)?;
        let tool_call_processor_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&tool_call_processor_config_json, 1)?
            .build()?;
        let _ = message.insert(
            tool_call_processor_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(tool_call_processor_config_table.get_name())
                .with_publisher("")
                .with_subject(tool_call_processor_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(tool_call_processor_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the dummy processor configs
        let processor_1_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            ..Default::default()
        };
        let processor_1_config_json = serde_json::to_vec(&processor_1_config)?;
        let processor_1_config_batches = create_bytes_record_batch(vec![processor_1_config_json])?;
        let processor_1_config_table = SubjectBuilder::new()
            .with_name("processor_1")
            .with_record_batches(vec![processor_1_config_batches])?
            .build()?;
        let _ = message.insert(
            processor_1_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_1_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_1_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_1_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor_2_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            ..Default::default()
        };
        let processor_2_config_json = serde_json::to_vec(&processor_2_config)?;
        let processor_2_config_batches = create_bytes_record_batch(vec![processor_2_config_json])?;
        let processor_2_config_table = SubjectBuilder::new()
            .with_name("processor_2")
            .with_record_batches(vec![processor_2_config_batches])?
            .build()?;
        let _ = message.insert(
            processor_2_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_2_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_2_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_2_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the mock subject_name table
        let task_names = vec!["task_1", "task_2", "task_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Join", "Filter"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["state_1", "processor_1"],
            vec!["state_2", "processor_2"],
            vec!["state_3", "processor_3"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"], vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["state_1"], vec!["state_2"], vec!["state_3"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| "session_1".to_string())
            .collect::<Vec<_>>();
        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
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

        // Create the processor and run
        let processor = ToolCallProcessor::new(name, ToolCallProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let table_reading = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream.remove(name).unwrap().message.take().unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, ["session_1", "session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, ["processor_1", "processor_2"]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, ["GroupBy", "Join"]);
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "OnUpdateAllRecordBatches",
                "AlwaysLastRecordBatch",
                "AlwaysAllRecordBatches",
                "OnUpdateAllRecordBatches",
                "AlwaysLastRecordBatch"
            ]
        );
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "state_1",
                "processor_1",
                "state_1",
                "state_2",
                "processor_2"
            ]
        );
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["Replace", "Replace"]);
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["state_1", "state_2"]);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_call_pipeline_from_struct() -> Result<()> {
        let name = "tool_call_pipeline";

        // Make the diagnostics and runtime env
        let span = SpanBuilder::default().with_span(name).build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Make the tool_call_processor config
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let tool_call_processor_config = ToolCallConfig {
            subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            subject_names: vec!["processor_1".to_string()],
            subscription_table_names: vec!["lhs_name".to_string(), "rhs_name".to_string()],
            ..Default::default()
        };
        let tool_call_processor_config_json = serde_json::to_vec(&tool_call_processor_config)?;
        let tool_call_processor_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&tool_call_processor_config_json, 1)?
            .build()?;
        let _ = message.insert(
            tool_call_processor_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(tool_call_processor_config_table.get_name())
                .with_publisher("")
                .with_subject(tool_call_processor_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(tool_call_processor_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the dummy processor configs
        let processor_1_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            ..Default::default()
        };
        let processor_1_config_json = serde_json::to_vec(&processor_1_config)?;
        let processor_1_config_table = SubjectBuilder::new()
            .with_name("processor_1")
            .with_json(&processor_1_config_json, 1)?
            .build()?;
        let _ = message.insert(
            processor_1_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_1_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_1_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_1_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor_2_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            ..Default::default()
        };
        let processor_2_config_json = serde_json::to_vec(&processor_2_config)?;
        let processor_2_config_table = SubjectBuilder::new()
            .with_name("processor_2")
            .with_json(&processor_2_config_json, 1)?
            .build()?;
        let _ = message.insert(
            processor_2_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_2_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_2_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_2_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the mock subject_name table
        let task_names = vec!["task_1", "task_1", "task_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Join", "Filter"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["AlwaysAllRecordBatches", "AlwaysLastRecordBatch"],
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
            vec!["OnUpdateAllRecordBatches", "AlwaysLastRecordBatch"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["state_1", "processor_1"],
            vec!["state_2", "processor_2"],
            vec!["state_3", "processor_3"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"], vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["state_1"], vec!["state_2"], vec!["state_3"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| "session_1".to_string())
            .collect::<Vec<_>>();
        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
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

        // Create the processor and run
        let processor = ToolCallProcessor::new(name, ToolCallProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let table_reading = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream.remove(name).unwrap().message.take().unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, ["session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, ["processor_1"]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, ["GroupBy"]);
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "AlwaysAllRecordBatches",
                "AlwaysLastRecordBatch",
            ]
        );
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "state_1",
                "processor_1",
            ]
        );
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["Replace"]);
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["state_1"]);

        Ok(())
    }
}
