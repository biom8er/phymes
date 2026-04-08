use anyhow::{Result, anyhow};
use arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use futures::{Stream, StreamExt};
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait,
    RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, SubjectBuilder, SubjectBuilderTrait, SubjectTrait
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, EventBuilderTrait, HashMap, MetricBuilderTrait,
};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_processor::ProcessorTrait;
use phymes_schemas::{create_bytes_fields, create_values_fields};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event, instrument};

use crate::{
    CandleTensorService, DataConfig, DataConfigTrait, DataOperatorTrait, DataStreamManager,
    TensorProcessorTrait, device,
};

/// Tensor processor made possible by Candle
///
/// Each operator has a defined input and output schema that calling processors or consuming processors
/// need to adhere to
#[derive(Debug)]
pub struct CandleDataProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for CandleDataProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for CandleDataProcessor {
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

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => {
                return Err(anyhow!(
                    "Config not provided for {}. Available options are {:?}.",
                    self.get_name(),
                    message.keys()
                ));
            }
        };

        // Run the ops
        let out = Box::pin(CandleDataStream::new(
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
    use crate::{DataDistanceOperator, candle_operators::AvailableCandleOperators};
    use arrow::array::{Float32Array, StringArray};
    use futures::TryStreamExt;
    use phymes_core::Subject;
    use phymes_diagnostics::{Diagnostics, SpanBuilder};
    use phymes_event::Publication;

    use super::*;

    #[tokio::test]
    async fn test_candle_ops_processor() -> Result<()> {
        // LHS and RHS messages
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec,
            lhs_embeddings_vec,
        )?;
        let lhs_table = Subject::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("lhs_name")
            .with_publisher("")
            .with_subject("lhs_name")
            .with_update(&Publication::None)
            .with_message(lhs_table.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs_batch = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec,
            rhs_embeddings_vec,
        )?;
        let rhs_table = Subject::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch])?
            .build()?;
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("rhs_name")
            .with_publisher("")
            .with_subject("rhs_name")
            .with_update(&Publication::None)
            .with_message(rhs_table.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);

        // Make the config
        let config = DataConfig {
            lhs_name: Some("lhs_name".to_string()),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: Some("lhs_pk".to_string()),
            lhs_fk: Some("lhs_fk".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_subject = SubjectBuilder::new()
            .with_name("candle_ops_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher("")
            .with_subject("candle_ops_processor")
            .with_update(&Publication::None)
            .with_message(config_subject.to_record_batch_stream())
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Make the stream and run
        let ops_processor = CandleDataProcessor::new("candle_ops_processor", "");
        let mut ops_stream =
            ops_processor.process(messages, Some(&diagnostic_builder), runtime_env)?;
        let result = ops_stream
            .remove("candle_ops_processor")
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        Ok(())
    }
}