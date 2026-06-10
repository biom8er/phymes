use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::OpenAIEmbedStream;
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use tracing::{Level, event};

use crate::ProcessorTrait;

#[derive(Debug)]
pub struct OpenAIEmbedProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for OpenAIEmbedProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for OpenAIEmbedProcessor {
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
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Re-index the messages by the subject name which needs to be unique at this stage
        let message = message
            .into_iter()
            .map(|(_k, v)| (v.get_subject().to_string(), v))
            .collect::<HashMap<_, _>>();

        // Run the embed stream
        let out = Box::pin(OpenAIEmbedStream::new(
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
    #[allow(unused_imports)]
    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
    #[allow(unused_imports)]
    use futures::TryStreamExt;
    #[allow(unused_imports)]
    use phymes_event::Publication;

    #[allow(unused_imports)]
    use super::*;

    #[cfg(not(feature = "candle"))]
    #[tokio::test]
    async fn test_openai_embed_processor() -> Result<()> {
        use phymes_diagnostics::{Diagnostics, SpanBuilder};

        use crate::AvailableOpenAIAssets;

        let config = CandleEmbedConfig {
            documents: "text".to_string(),
            input_type: "passage".to_string(),
            api_url: Some("http://0.0.0.0:8001/v1".to_string()),
            openai_asset: Some(AvailableOpenAIAssets::NvidiaLlamaV3p2NvEmbedQA1BV2),
            ..Default::default()
        };

        // Make the config
        let config_table = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config.clone())?, 1)?
            .build()?;

        // Make the runtime
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Case 1: streaming query
        // Make the query input stream
        let query_vec = vec![
            "How much protein should a female eat.",
            "Summit define",
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        ];
        let text: ArrayRef = Arc::new(StringArray::from(query_vec));
        let batch = RecordBatch::try_from_iter(vec![("text", text)])?;
        let document_table = SubjectBuilder::new()
            .with_name("text")
            .with_record_batches(vec![batch])?
            .build()?;
        let document_message = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher("")
            .with_subject("text")
            .with_update(&Publication::None)
            .with_message(document_table.to_record_batch_stream())
            .make_name()?
            .build()?;
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(document_message.get_name().to_string(), document_message);

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make and run the embeddings stream
        let embed_stream = OpenAIEmbedStream::new(
            messages,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder),
        )?;
        let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
        assert_eq!(embeddings.len(), 1);

        // Expected data
        let embeddings_test: Vec<Vec<f32>> = vec![
            vec![-0.0199745, -0.03612664, 0.015255524],
            vec![0.013020273, 0.012949716, 0.015651252],
            vec![-0.016750623, 0.017388858, -0.007890748],
            vec![0.038596537, 0.00942193, 0.011650219],
        ];
        let embeddings_vec = embeddings
            .first()
            .unwrap()
            .column_by_name("embeddings")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .iter()
            .map(|s| {
                s.unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|f| f.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            embeddings_vec.first().unwrap()[0..3],
            embeddings_test.first().unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(1).unwrap()[0..3],
            embeddings_test.get(1).unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(2).unwrap()[0..3],
            embeddings_test.get(2).unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(3).unwrap()[0..3],
            embeddings_test.get(3).unwrap()[0..3]
        );

        // Case 2: streaming query with multiple batches
        // Make the query input stream
        let query_vec1 = vec!["How much protein should a female eat.", "Summit define"];
        let query_vec2 = vec![
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        ];
        let embeddings1: ArrayRef = Arc::new(StringArray::from(query_vec1));
        let embeddings2: ArrayRef = Arc::new(StringArray::from(query_vec2));
        let batch1 = RecordBatch::try_from_iter(vec![("text", embeddings1)])?;
        let batch2 = RecordBatch::try_from_iter(vec![("text", embeddings2)])?;
        let document_table = SubjectBuilder::new()
            .with_name("text")
            .with_record_batches(vec![batch1, batch2])?
            .build()?;
        let document_message = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher("")
            .with_subject("text")
            .with_update(&Publication::None)
            .with_message(document_table.to_record_batch_stream())
            .make_name()?
            .build()?;
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(document_message.get_name().to_string(), document_message);

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make and run the embeddings stream
        let embed_stream = OpenAIEmbedStream::new(
            messages,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder),
        )?;
        let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
        assert_eq!(embeddings.len(), 2);
        let embeddings_vec = embeddings
            .first()
            .unwrap()
            .column_by_name("embeddings")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .iter()
            .map(|s| {
                s.unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|f| f.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Expected data
        let embeddings_test: Vec<Vec<f32>> = vec![
            vec![-0.019996276, -0.036101155, 0.015297651],
            vec![0.013022803, 0.012991025, 0.015626218],
        ];
        assert_eq!(
            embeddings_vec.first().unwrap()[0..3],
            embeddings_test.first().unwrap()[0..3]
        );
        assert_eq!(
            embeddings_vec.get(1).unwrap()[0..3],
            embeddings_test.get(1).unwrap()[0..3]
        );
        Ok(())
    }
}
