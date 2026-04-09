use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::CoalesceStream;

use crate::ProcessorTrait;

/// Processor that implements the [RecordBatch] coalesce operator to combine smaller [RecordBatch]es into larger [RecordBatch]es of a specified size
#[derive(Debug)]
pub struct CoalesceProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for CoalesceProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for CoalesceProcessor {
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
        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Extract out the message
        let mut subscriptions = message.into_values().collect::<Vec<_>>();
        if subscriptions.len() > 1 {
            return Err(anyhow!(
                "More than one subscription was found for {}.",
                self.get_name()
            ));
        } else if subscriptions.is_empty() {
            return Err(anyhow!(
                "No subscriptions were found for {}.",
                self.get_name()
            ));
        }

        // Run the coalesce stream
        let out = Box::pin(CoalesceStream::new(
            subscriptions.swap_remove(0).get_message_own(),
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.cloned(),
        ));

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

    use phymes_subject::{SubjectBuilder, SubjectBuilderTrait, SubjectTrait, test_subject};
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_event::Publication;
    use phymes_streams::LimitConfig;

    #[tokio::test]
    async fn test_coalesce_processor() -> Result<()> {
        // Make the test batches
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let test_table = test_subject::make_test_subject("input", 4, 8, 4)?;
        let test_message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("input")
            .with_subject("input")
            .with_publisher("")
            .with_message(test_table.to_record_batch_stream())
            .with_update(&Publication::None)
            .build()?;
        let _ = message.insert(test_message.get_name().to_string(), test_message);

        // Make the config
        let config = LimitConfig {
            fetch: 6,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("CoalesceProcessor")
            .with_json(&config_json, 1)?
            .build()?;
        let config_message = SendableRecordBatchStreamMessage::get_builder()
            .with_name(config_table.get_name())
            .with_publisher("")
            .with_subject(config_table.get_name())
            .with_update(&Publication::None)
            .with_message(config_table.to_record_batch_stream())
            .build()?;
        let _ = message.insert(config_message.get_name().to_string(), config_message);

        // Make the diagnostics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the Runtime Env
        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            ..Default::default()
        });

        // Coalesce into batches of six
        let processor = CoalesceProcessor::new("CoalesceProcessor", "");
        let mut stream =
            processor.process(message, Some(&diagnostic_builder), runtime_env.clone())?;

        // Wrap the results in a table
        let partitions = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream
                .remove("CoalesceProcessor")
                .unwrap()
                .message
                .take()
                .unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        let sizes = partitions
            .get_record_batches()
            .iter()
            .map(|b| b.num_rows())
            .collect::<Vec<_>>();
        assert_eq!(sizes, [6, 6, 4]);
        Ok(())
    }
}
