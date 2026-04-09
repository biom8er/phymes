use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::TokenStreamTrait;
use phymes_streams::CandleEmbedStream;
use tracing::{Level, event, instrument};

use crate::{ProcessorTrait, TokenStreamTraitExt};

#[derive(Debug)]
pub struct CandleEmbedProcessor {
    name: String,
    r#type: String,
    token_service: Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>>,
}

impl MappableTrait for CandleEmbedProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for CandleEmbedProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
            token_service: Arc::new(Mutex::new(None)),
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
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Re-index the messages by the subject name which needs to be unique at this stage
        let message = message
            .into_iter()
            .map(|(_k, v)| (v.get_subject().to_string(), v))
            .collect::<HashMap<_, _>>();

        // run the embed stream
        let out = Box::pin(CandleEmbedStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            Arc::clone(self.token_service()),
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

impl TokenStreamTraitExt for CandleEmbedProcessor {
    fn token_service(&self) -> &Arc<Mutex<Option<Box<dyn TokenStreamTrait>>>> {
        &self.token_service
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Float32Array, ListArray, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use phymes_subject::{Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_event::Publication;
    use phymes_ml::{AvailableCandleAssets, CandleEmbedConfig};

    use super::*;

    #[tokio::test]
    async fn test_candle_embed_stream_nowasm() -> Result<()> {
        let config = CandleEmbedConfig {
            documents: "text".to_string(),
            // WASM testing
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(AvailableCandleAssets::QwenV2_1p5bEmbed),
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
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat.",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define",
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
        let embed_stream = CandleEmbedStream::new(
            messages,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Arc::new(Mutex::new(None)),
            Some(diagnostic_builder.clone()),
        )?;

        // DM: Skip actually running the tests as they take too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
            assert_eq!(embeddings.len(), 1);

            // Expected data
            let _embeddings_test: Vec<Vec<f32>> = vec![
                vec![-3.0385482, 7.2247167, 3.2304974],
                vec![2.4326377, 1.8344411, -0.7329114],
                vec![-2.7296476, 6.784784, 3.0706217],
                vec![-4.392374, 2.938572, -4.162841],
            ];
            let _embeddings_vec = embeddings
                .first()
                .unwrap()
                .column_by_name("embedding")
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
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

            // DM: the results also dependent upon the system the model is ran
            // assert_eq!(
            //     embeddings_vec.first().unwrap()[0..3],
            //     embeddings_test.first().unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(1).unwrap()[0..3],
            //     embeddings_test.get(1).unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(2).unwrap()[0..3],
            //     embeddings_test.get(2).unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(3).unwrap()[0..3],
            //     embeddings_test.get(3).unwrap()[0..3]
            // );

            // Case 2: streaming query with multiple batches
            // Make the query input stream
            let query_vec1 = vec![
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat.",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define",
            ];
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

            // Make and run the embeddings stream
            let embed_stream = CandleEmbedStream::new(
                messages,
                config_table.to_record_batch_stream(),
                Arc::clone(&runtime_env),
                Arc::new(Mutex::new(None)),
                Some(diagnostic_builder.clone()),
            )?;
            let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
            assert_eq!(embeddings.len(), 2);
            let embeddings_vec = embeddings
                .first()
                .unwrap()
                .column_by_name("embedding")
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
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
            let _embeddings_test: Vec<Vec<f32>> = vec![
                vec![-3.29946, 7.5989823, 3.311682],
                vec![2.2690444, 2.090072, -0.8259398],
            ];
            assert_eq!(embeddings_vec.first().unwrap().len(), 1536); // hidden size in config.json

            // DM: the results also dependent upon the system the model is ran
            // assert_eq!(
            //     embeddings_vec.first().unwrap()[0..3],
            //     embeddings_test.first().unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(1).unwrap()[0..3],
            //     embeddings_test.get(1).unwrap()[0..3]
            // );
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_candle_embed_stream_wasm() -> Result<()> {
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

        // Make the config
        let config = CandleEmbedConfig {
            documents: "text".to_string(),
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                // "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin",
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                // crate::candle_assets::candle_which::WhichCandleAsset::BertEmbed,
                AvailableCandleAssets::QuantizedBertEmbed,
            ),
            ..Default::default()
        };
        let config_table = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Make and run the embeddings stream
        let embed_stream = CandleEmbedStream::new(
            messages,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Arc::new(Mutex::new(None)),
            Some(diagnostic_builder.clone()),
        )?;

        // DM: Skip actually running the tests as they take too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            let embeddings = embed_stream.try_collect::<Vec<_>>().await?;
            assert_eq!(embeddings.len(), 1);

            // Expected data
            let _embeddings_test: Vec<Vec<f32>> = vec![
                vec![-3.2244308, 7.4192524, 2.9019766],
                vec![2.163365, 1.8837537, -0.18565525],
                vec![-3.260014, 6.5834556, 2.9206438],
                vec![-5.446545, 2.0517492, -4.0273705],
            ];
            let _embeddings_vec = embeddings
                .first()
                .unwrap()
                .column_by_name("embedding")
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
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

            // DM: the results also dependent upon the system the model is ran
            // assert_eq!(
            //     embeddings_vec.first().unwrap()[0..3],
            //     embeddings_test.first().unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(1).unwrap()[0..3],
            //     embeddings_test.get(1).unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(2).unwrap()[0..3],
            //     embeddings_test.get(2).unwrap()[0..3]
            // );
            // assert_eq!(
            //     embeddings_vec.get(3).unwrap()[0..3],
            //     embeddings_test.get(3).unwrap()[0..3]
            // );
        }
        Ok(())
    }
}
