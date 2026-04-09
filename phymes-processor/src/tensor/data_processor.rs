use anyhow::{Result, anyhow};
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::CandleDataStream;
use std::sync::Arc;
use tracing::{Level, event, instrument};

use crate::ProcessorTrait;

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
    use arrow::array::{Float32Array, StringArray};
    use futures::TryStreamExt;
    use phymes_subject::{Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
    use phymes_data::{
        AvailableOperators, DataConfig, DataDistanceOperator, DataStreamManager, test_candle_ops,
    };
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_event::Publication;

    use super::*;

    #[tokio::test]
    async fn test_candle_ops_stream() -> Result<()> {
        // Case 1:  LHS and RHS messages from single stream batch
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = test_candle_ops::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec,
            lhs_embeddings_vec,
        )?;
        let lhs_table = Subject::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs_batch = test_candle_ops::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec,
            rhs_embeddings_vec,
        )?;
        let rhs_table = Subject::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch])?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

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
            operator: AvailableOperators::VectorDistance,
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

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

        // Case 2: LHS and RHS from config

        // Make the config
        let config_args = DataConfig {
            operator: AvailableOperators::HumanInTheLoop,
            lhs_name: Some("".to_string()),
            lhs_args: Some("{\"role\": \"assistant\", \"content\": \"RESPONSE\"}".to_string()),
            rhs_args: None,
            ..Default::default()
        };
        let config_args_table = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config_args)?, 1)?
            .build()?;

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            HashMap::<String, SendableRecordBatchStreamMessage>::new(),
            config_args_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("role")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id.first().unwrap(), &"assistant");
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("content")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id.first().unwrap(), &"RESPONSE");

        // Case 3: LHS and RHS messages from multiple stream batch (accumulate LHS and RHS)
        let lhs_ids_vec_1 = vec!["1"];
        let lhs_embeddings_vec_1: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.]];
        let lhs_batch_1 = test_candle_ops::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec_1,
            lhs_embeddings_vec_1,
        )?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_embeddings_vec_2: Vec<Vec<f32>> = vec![vec![0., 1., 0., 1.], vec![0., 0., 0., 1.]];
        let lhs_batch_2 = test_candle_ops::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec_2,
            lhs_embeddings_vec_2,
        )?;
        let lhs_table = Subject::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch_1, lhs_batch_2])?
            .build()?;
        let rhs_ids_vec_1 = vec!["1", "2"];
        let rhs_embeddings_vec_1: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.], vec![1., 1., 1., 1.]];
        let rhs_batch_1 = test_candle_ops::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec_1,
            rhs_embeddings_vec_1,
        )?;
        let rhs_ids_vec_2 = vec!["3", "4"];
        let rhs_embeddings_vec_2: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.], vec![1., 1., 1., 1.]];
        let rhs_batch_2 = test_candle_ops::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec_2,
            rhs_embeddings_vec_2,
        )?;
        let rhs_table = Subject::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch_1, rhs_batch_2])?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

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

        // Case 4: LHS and RHS messages from multiple stream batch (accumulate LHS and Stream RHS)
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
            operator: AvailableOperators::VectorDistance,
            lhs_stream: DataStreamManager::Accumulate,
            rhs_stream: Some(DataStreamManager::Stream),
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values (for RHS streaming)
        let lhs_ids_test = vec!["1", "1", "2", "2", "3", "3", "1", "1", "2", "2", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "1", "2", "1", "2", "3", "4", "3", "4", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 0.70710677, 0.70710677, 0.5, 0.5, 1.0, 1.0, 0.70710677, 0.70710677, 0.5, 0.5,
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

        // Case 5: LHS and RHS messages from multiple stream batch (Stream LHS and Stream RHS)
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
            operator: AvailableOperators::VectorDistance,
            lhs_stream: DataStreamManager::Stream,
            rhs_stream: Some(DataStreamManager::Stream),
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "2", "2", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "3", "4"];
        let scores_test: Vec<f32> = vec![1.0, 1.0, 0.70710677, 0.70710677, 0.5, 0.5];

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

        // Case 6: LHS and RHS messages from multiple stream batch (Stream LHS and Accumulate RHS)
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
            operator: AvailableOperators::VectorDistance,
            lhs_stream: DataStreamManager::Stream,
            rhs_stream: Some(DataStreamManager::Accumulate),
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

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

    #[tokio::test]
    async fn test_candle_ops() -> Result<()> {
        // LHS and RHS messages
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = test_candle_ops::make_embeddings_record_batch_str_f32(
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
        let rhs_batch = test_candle_ops::make_embeddings_record_batch_str_f32(
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
            operator: AvailableOperators::VectorDistance,
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
